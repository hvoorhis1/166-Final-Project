# Redox Descriptor Pipeline (Torch Version)

This README is the quick run/reproduce guide for the current torch-based pipeline.

## What This Version Does
- Uses descriptor tables only (no structure featurization).
- Uses two feature sets:
  - `X_full`
  - `X_reduced` (top target-correlation descriptors)
- Trains/tunes three PyTorch model families:
  - `TorchLinear`
  - `TorchMLPSmall`
  - `TorchMLPDeep`
- Uses 5-fold CV for selection and held-out test evaluation.
- Produces exactly 12 key plots with torch-accurate names.

## Run

```bash
cd /Users/harryvoorhis/Documents/Playground
PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mplconfig ./.venv/bin/python project/main.py
```

## Main Outputs

### Tables
- `project/results/model_comparison.csv`
- `project/results/feature_ranking.csv`
- `project/results/screening_results.csv`
- `project/data/preprocessing_log.csv`

### Plots (12)
- `project/plots/target_distribution.png`
- `project/plots/cv_torch_linear.png`
- `project/plots/cv_torch_mlp_small.png`
- `project/plots/cv_torch_mlp_deep.png`
- `project/plots/torch_linear_test_scatter.png`
- `project/plots/torch_mlp_small_test_scatter.png`
- `project/plots/torch_mlp_deep_test_scatter.png`
- `project/plots/model_comparison_r2.png`
- `project/plots/model_comparison_rmse.png`
- `project/plots/predicted_vs_actual.png`
- `project/plots/residuals_vs_predicted.png`
- `project/plots/screening_scatter.png`

## Full Documentation
For full methods, model-choice rationale, and plot interpretation details, see:
- `project/PIPELINE_REPORT.md`
