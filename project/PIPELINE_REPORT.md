# Descriptor-Based Redox Pipeline Report (Torch)

## Scope
This pipeline predicts quinone redox potential (`deltaE_V`) from precomputed molecular descriptors (no RDKit feature generation).

- ID column: `Name`
- Target: `deltaE_V`
- Script: `project/main.py`

## Why These Torch Models
The model set is intentionally minimal and complementary:

1. `TorchLinear`
- Purpose: strong linear baseline with L2 regularization via weight decay.
- Why included: high interpretability and fast training; establishes a lower-complexity reference.

2. `TorchMLPSmall` (1 hidden layer)
- Purpose: capture moderate nonlinear relationships with low parameter count.
- Why included: usually best tradeoff between flexibility and overfitting risk for medium tabular datasets.

3. `TorchMLPDeep` (2 hidden layers)
- Purpose: higher-capacity nonlinear model.
- Why included: tests whether added depth materially improves generalization beyond `TorchMLPSmall`.

Together, these three models provide a clean capacity ladder: linear -> shallow nonlinear -> deeper nonlinear.

## Data and Preprocessing
- Uses predefined train/test split files when present.
- Descriptor cleaning on training data context:
  - numeric-only features
  - remove all-missing columns
  - remove zero-variance columns
  - remove one feature from each high-correlation pair (`|r| > 0.95`)
- Feature sets:
  - `X_full`: all cleaned descriptors
  - `X_reduced`: top descriptors by absolute correlation with target
- Preprocessing is leakage-safe:
  - median imputation fit on train folds only
  - z-score scaling fit on train folds only

## Training and Selection
- 5-fold CV on training data for each model-feature-set combination.
- Hyperparameter grids:
  - `TorchLinear`: `weight_decay`
  - `TorchMLPSmall` / `TorchMLPDeep`: hidden width, dropout, weight decay (with fixed LR/epochs per grid point)
- Optimizer/loss: Adam + MSE.
- Best config selected by mean CV R².
- Final metrics reported on held-out test set.

## Latest Run Summary
From `project/results/model_comparison.csv`:

- Best: **TorchMLPSmall on X_full**
  - Test RMSE: **0.1164**
  - Test MAE: **0.0837**
  - Test R²: **0.9316**

Other tested configs:
- TorchMLPDeep on X_reduced: Test R² 0.8916
- TorchMLPDeep on X_full: Test R² 0.8887
- TorchLinear on X_full: Test R² 0.8342

## Screening Summary
From `project/results/screening_results.csv`:
- Passes all filters: **379** molecules
- Top passing candidate by predicted redox potential: `12754795_b3lyp_1`

## Plot Interpretation Guide (Accurate Filenames)
The current pipeline outputs exactly 12 plots.

1. `target_distribution.png`
- What: target histogram with mean/median lines.
- Why: shows task distribution shape and central tendency.

2. `cv_torch_linear.png`
- What: CV R² vs `weight_decay` for `TorchLinear`.
- Why: diagnoses under/over-regularization in the linear baseline.

3. `cv_torch_mlp_small.png`
- What: CV R² trend vs the key tuning axis for `TorchMLPSmall`.
- Why: indicates the best small-MLP capacity regime.

4. `cv_torch_mlp_deep.png`
- What: CV R² trend vs the key tuning axis for `TorchMLPDeep`.
- Why: evaluates if extra depth improves generalization robustly.

5. `torch_linear_test_scatter.png`
- What: predicted vs actual scatter for best linear config.
- Why: linear calibration/bias view.

6. `torch_mlp_small_test_scatter.png`
- What: predicted vs actual for best small MLP.
- Why: nonlinear calibration check for the best-performing family.

7. `torch_mlp_deep_test_scatter.png`
- What: predicted vs actual for best deep MLP.
- Why: compares deeper capacity against small MLP behavior.

8. `model_comparison_r2.png`
- What: grouped bar chart of test R².
- Why: fast ranking by explained variance.

9. `model_comparison_rmse.png`
- What: grouped bar chart of test RMSE.
- Why: compares absolute error in target units.

10. `predicted_vs_actual.png`
- What: canonical best-model test scatter.
- Why: primary paper-ready fit visualization.

11. `residuals_vs_predicted.png`
- What: residual pattern for best model.
- Why: checks bias trends and heteroscedasticity.

12. `screening_scatter.png`
- What: predicted redox potential vs capacity proxy, colored by `passes_all`.
- Why: final decision-space visualization for candidate prioritization.

## Output Files
- `project/main.py`
- `project/data/preprocessing_log.csv`
- `project/results/model_comparison.csv`
- `project/results/feature_ranking.csv`
- `project/results/screening_results.csv`
- `project/plots/*.png` (12 files listed above)
