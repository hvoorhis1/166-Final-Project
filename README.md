# Machine Learning Screening of Organic Cathode Molecules

This repository contains code to train and evaluate machine learning models for redox potential prediction from molecular structure, with lightweight screening heuristics for practical candidate triage.

## What this project includes

- Data loading and validation for tabular molecular datasets.
- RDKit featurization from SMILES:
  - descriptor-based features,
  - Morgan fingerprints,
  - hybrid descriptor + fingerprint features.
- Regression baselines:
  - Ridge Regression,
  - Random Forest,
  - Gradient Boosting.
- Evaluation with train/test split and cross-validation metrics.
- Simple screening heuristics for practical constraints (size and dissolution risk proxies).
- Saved outputs: metrics, feature importances, predictions, and ranked candidates.

## Expected dataset format

Provide either:

- a CSV with SMILES and target columns, or
- the `org-redox-dataset-main` directory (auto-loads `datasets/DatasetQuinonesFiltered.csv`).

Default target is `deltaE_V`. If a `smiles` column is present, the pipeline uses descriptor/fingerprint/hybrid featurization from SMILES. If not, it automatically trains from the precomputed numeric descriptors.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you want SMILES-based featurization, install with RDKit extra:

```bash
pip install -e .[smiles]
```

## Run training and screening

```bash
redox-train \
  --data-path /Users/harryvoorhis/Downloads/org-redox-dataset-main \
  --output-dir results/run1 \
  --smiles-col smiles \
  --target-col deltaE_V
```

Optional arguments:

- `--fingerprint-bits 2048`
- `--fingerprint-radius 2`
- `--test-size 0.2`
- `--random-state 42`
- `--n-jobs -1`

## Outputs

The run directory will contain:

- `metrics.csv`: model/representation performance summary.
- `cv_metrics.csv`: cross-validation score summary.
- `predictions_<representation>_<model>.csv`: per-molecule predictions for test split.
- `top_candidates_<representation>_<model>.csv`: highest predicted-potential molecules after heuristic flags.
- `feature_importance_descriptor_random_forest.csv`: descriptor importances for interpretability.

## Notes

- Heuristic flags are simple proxies and not replacements for mechanistic simulations.
- For publication-quality results, tune hyperparameters and benchmark against stronger models.
