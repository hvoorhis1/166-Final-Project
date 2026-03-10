from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


DESCRIPTOR_REGISTRY: dict[str, callable] = {
    "mw": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "hbd": Descriptors.NumHDonors,
    "hba": Descriptors.NumHAcceptors,
    "rot_bonds": Descriptors.NumRotatableBonds,
    "ring_count": Descriptors.RingCount,
    "aromatic_rings": Descriptors.NumAromaticRings,
    "heteroatoms": Descriptors.NumHeteroatoms,
    "fraction_csp3": Descriptors.FractionCSP3,
}


@dataclass(frozen=True)
class FingerprintConfig:
    radius: int = 2
    n_bits: int = 2048


def smiles_to_mols(smiles: pd.Series) -> list[Chem.Mol]:
    mols = [Chem.MolFromSmiles(s) for s in smiles.astype(str)]
    if any(m is None for m in mols):
        raise ValueError("Encountered invalid molecule in featurization step")
    return mols


def build_descriptor_matrix(mols: list[Chem.Mol]) -> tuple[np.ndarray, list[str]]:
    names = list(DESCRIPTOR_REGISTRY.keys())
    rows = []
    for mol in mols:
        rows.append([DESCRIPTOR_REGISTRY[name](mol) for name in names])
    return np.asarray(rows, dtype=float), names


def build_fingerprint_matrix(
    mols: list[Chem.Mol],
    cfg: FingerprintConfig,
) -> tuple[np.ndarray, list[str]]:
    generator = GetMorganGenerator(radius=cfg.radius, fpSize=cfg.n_bits)
    mat = np.zeros((len(mols), cfg.n_bits), dtype=np.float32)
    for i, mol in enumerate(mols):
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((cfg.n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        mat[i, :] = arr
    feature_names = [f"fp_{i}" for i in range(cfg.n_bits)]
    return mat, feature_names


def build_feature_sets(
    smiles: pd.Series,
    fp_cfg: FingerprintConfig,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    mols = smiles_to_mols(smiles)
    x_desc, desc_names = build_descriptor_matrix(mols)
    x_fp, fp_names = build_fingerprint_matrix(mols, fp_cfg)
    x_hybrid = np.hstack([x_desc, x_fp])
    hybrid_names = [*desc_names, *fp_names]

    return {
        "descriptor": (x_desc, desc_names),
        "fingerprint": (x_fp, fp_names),
        "hybrid": (x_hybrid, hybrid_names),
    }
