from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional path for precomputed-only usage
    Chem = None


@dataclass(frozen=True)
class DatasetSpec:
    smiles_col: str = "smiles"
    target_col: str = "deltaE_V"
    id_col: str = "Name"


@dataclass(frozen=True)
class LoadedDataset:
    df: pd.DataFrame
    target_col: str
    smiles_col: str | None
    id_col: str | None
    mode: str


def load_dataset(path: str | Path, spec: DatasetSpec) -> LoadedDataset:
    path = _resolve_dataset_path(path)
    df = pd.read_csv(path)

    target_col = _resolve_target_col(df, spec.target_col)
    smiles_col = _resolve_smiles_col(df, spec.smiles_col)
    id_col = spec.id_col if spec.id_col in df.columns else None

    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    if smiles_col is not None:
        df = df.dropna(subset=[smiles_col]).copy()
        valid_mask = df[smiles_col].astype(str).apply(_is_valid_smiles)
        dropped = int((~valid_mask).sum())
        if dropped:
            print(f"Dropped {dropped} rows with invalid SMILES")
        df = df[valid_mask].reset_index(drop=True)
        mode = "smiles"
    else:
        mode = "precomputed"

    return LoadedDataset(
        df=df,
        target_col=target_col,
        smiles_col=smiles_col,
        id_col=id_col,
        mode=mode,
    )


def build_precomputed_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    id_col: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    exclude = {target_col}
    if id_col is not None:
        exclude.add(id_col)

    numeric_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        raise ValueError("No numeric feature columns were found for precomputed mode.")

    x = df[numeric_cols].to_numpy(dtype=float)
    return x, numeric_cols


def _resolve_dataset_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.is_file():
        return path

    candidate_names = [
        "DatasetQuinonesFiltered.csv",
        "DatasetQuinonesFilteredTrain.csv",
        "DatasetQuinonesFilteredTest.csv",
    ]
    for name in candidate_names:
        p = path / "datasets" / name
        if p.exists():
            return p

    csvs = sorted(path.glob("*.csv"))
    if csvs:
        return csvs[0]

    raise FileNotFoundError(
        f"No CSV file found at {path}. Pass a CSV path or dataset directory."
    )


def _resolve_target_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred

    fallbacks = ["deltaE_V", "redox_potential", "target", "y"]
    for col in fallbacks:
        if col in df.columns:
            print(f"Target column '{preferred}' not found. Using '{col}' instead.")
            return col

    raise ValueError(
        "Could not find a valid target column. "
        f"Tried '{preferred}' and fallbacks {fallbacks}."
    )


def _resolve_smiles_col(df: pd.DataFrame, preferred: str) -> str | None:
    if preferred in df.columns:
        return preferred
    for col in df.columns:
        if col.lower() == "smiles":
            return col
    return None


def _is_valid_smiles(smiles: str) -> bool:
    if Chem is None:
        raise ImportError("RDKit is required for SMILES-based featurization.")
    return Chem.MolFromSmiles(smiles) is not None
