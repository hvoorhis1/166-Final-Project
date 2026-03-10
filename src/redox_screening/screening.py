from __future__ import annotations

import pandas as pd
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski
except ImportError:  # pragma: no cover - optional path when using precomputed descriptors
    Chem = None
    Crippen = None
    Descriptors = None
    Lipinski = None


def compute_screening_flags(df: pd.DataFrame, smiles_col: str | None = None) -> pd.DataFrame:
    if smiles_col is not None and smiles_col in df.columns:
        if Chem is None:
            raise ImportError("RDKit is required for SMILES-based screening heuristics.")
        return _compute_screening_flags_from_smiles(df[smiles_col]).reset_index(drop=True)
    return _compute_screening_flags_from_columns(df).reset_index(drop=True)


def _compute_screening_flags_from_smiles(smiles: pd.Series) -> pd.DataFrame:
    rows = []
    for s in smiles.astype(str):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            rows.append(
                {
                    "mw": float("nan"),
                    "logp": float("nan"),
                    "hba": float("nan"),
                    "hbd": float("nan"),
                    "tpsa": float("nan"),
                    "flag_large_molecule": True,
                    "flag_dissolution_risk": True,
                }
            )
            continue

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hba = Lipinski.NumHAcceptors(mol)
        hbd = Lipinski.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)

        # Practical proxy thresholds for early triage.
        flag_large = mw > 450
        flag_dissolution = (logp < 0.5) and (tpsa > 80 or (hba + hbd) >= 8)

        rows.append(
            {
                "mw": mw,
                "logp": logp,
                "hba": hba,
                "hbd": hbd,
                "tpsa": tpsa,
                "flag_large_molecule": flag_large,
                "flag_dissolution_risk": flag_dissolution,
            }
        )

    return pd.DataFrame(rows)


def _compute_screening_flags_from_columns(df: pd.DataFrame) -> pd.DataFrame:
    mw = _first_numeric(df, ["MW", "mw", "MolWt"])
    logp = _first_numeric(df, ["logp", "LogP", "MLogP"])
    hba = _first_numeric(df, ["nHBAcc_Lipinski", "nHBAcc", "HBA", "hba"])
    hbd = _first_numeric(df, ["nHBD", "HBD", "hbd"])
    tpsa = _first_numeric(df, ["TopoPSA", "TPSA", "tpsa"])

    # If a proxy is unavailable, avoid hard-flagging and leave metric as NaN.
    flag_large = mw > 450 if mw is not None else pd.Series(False, index=df.index)

    if logp is not None and tpsa is not None and hba is not None:
        hbd_for_sum = hbd.fillna(0.0) if hbd is not None else 0.0
        flag_dissolution = (logp < 0.5) & ((tpsa > 80) | ((hba + hbd_for_sum) >= 8))
    else:
        flag_dissolution = pd.Series(False, index=df.index)

    return pd.DataFrame(
        {
            "mw": mw if mw is not None else float("nan"),
            "logp": logp if logp is not None else float("nan"),
            "hba": hba if hba is not None else float("nan"),
            "hbd": hbd if hbd is not None else float("nan"),
            "tpsa": tpsa if tpsa is not None else float("nan"),
            "flag_large_molecule": flag_large.astype(bool),
            "flag_dissolution_risk": flag_dissolution.astype(bool),
        }
    )


def _first_numeric(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return None


def apply_candidate_filter(df: pd.DataFrame) -> pd.DataFrame:
    large = df["flag_large_molecule"].fillna(False).astype(bool)
    dissolution = df["flag_dissolution_risk"].fillna(False).astype(bool)
    return df[(~large) & (~dissolution)].copy()
