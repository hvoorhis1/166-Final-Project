# Machine Learning Screening of Organic Cathode Molecules via Redox Potential Prediction

CHEMENG 177 / MATSCI 166 — Winter 2026 Final Project

Vi Cheng, Harry Voorhis, Brian Donohugh — Stanford University

## Overview

This project develops machine learning models to predict the redox potential (`deltaE_V`) of quinone-based organic molecules from precomputed molecular descriptors. We compare Ridge regression, Random Forest, and a small multilayer perceptron (MLP) against physics-informed baselines built from frontier orbital energies (LUMO, HOMO). The best predictor is embedded into a multi-objective screening workflow that ranks candidate cathode molecules by predicted redox potential and theoretical gravimetric capacity.

## Repository Structure

```
├── project/
│   ├── main.py                # Full pipeline: preprocessing, training, evaluation, screening
│   ├── data/                  # Dataset CSVs (included)
│   │   ├── DatasetQuinonesFiltered.csv
│   │   ├── DatasetQuinonesFilteredTrain.csv
│   │   └── DatasetQuinonesFilteredTest.csv
│   ├── plots/                 # Generated figures (created by main.py)
│   └── results/               # Generated CSV results (created by main.py)
├── requirements.txt
└── README.md
```

## Data

This project uses the [org-redox-dataset](https://github.com/ersilia-os/org-redox-dataset), a publicly available molecular database of small redox-active organic molecules with B3LYP-optimized geometries and tabulated molecular descriptors.

The required CSV files are included in `project/data/`:

| File | Description |
|------|-------------|
| `DatasetQuinonesFiltered.csv` | Full dataset (494 molecules, 221 columns) |
| `DatasetQuinonesFilteredTrain.csv` | Pre-defined training split (395 molecules) |
| `DatasetQuinonesFilteredTest.csv` | Pre-defined test split (99 molecules) |

The regression target is `deltaE_V` (redox potential in volts). Input features are precomputed numeric molecular descriptors covering electronic, topological, charge, and physicochemical properties.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/<your-username>/166-Final-Project.git
cd 166-Final-Project
pip install -r requirements.txt
```

## Running

Run the full pipeline from the repository root:

```bash
cd project
python main.py
```

This executes the entire workflow:

1. **Data loading and EDA** — loads train/test splits, prints dataset summary statistics, plots target distribution
2. **Feature cleaning** — removes constant, all-missing, and highly correlated (|r| > 0.95) descriptors
3. **Model training** — grid-search with 5-fold cross-validation over Ridge, Random Forest, and MLP hyperparameters
4. **Physics baselines** — evaluates compact models using only LUMO, LUMO+HOMO, and a small electronic descriptor set
5. **Descriptor-family ablation** — trains Ridge on each descriptor family independently
6. **Evaluation** — reports test R², RMSE, MAE with 95% bootstrap confidence intervals
7. **Screening** — ranks all molecules by a multi-objective score combining predicted redox potential and capacity proxy, identifies Pareto-optimal candidates

### Optional: external data directory

If the dataset CSVs are stored elsewhere, pass the path:

```bash
python main.py --data-dir /path/to/org-redox-dataset/datasets
```

## Output

After running, the pipeline produces:

**`project/results/`**:
- `model_comparison.csv` — all experiment results (metrics per model × feature set)
- `results_with_ci.csv` — test metrics with 95% bootstrap confidence intervals
- `feature_ranking.csv` — descriptor importance ranking from the best model
- `descriptor_family_performance.csv` — per-family ablation results
- `screening_results.csv` — full screening table for all 494 molecules
- `top20_candidates.csv` — top 20 candidates by screening score

**`project/plots/`**:
- `target_distribution.png` — histogram of deltaE_V
- `model_comparison_r2.png`, `model_comparison_rmse.png` — bar charts comparing all models
- `physics_vs_ml_r2.png`, `physics_vs_ml_rmse.png` — physics baselines vs full ML models
- `descriptor_family_performance.png` — Ridge performance per descriptor family
- `high_potential_performance.png` — RMSE in the top 20% regime
- `cv_ridge.png`, `cv_random_forest.png`, `cv_torch_mlp_small.png` — CV tuning curves
- `ridge_test_scatter.png`, `random_forest_test_scatter.png`, `torch_mlp_small_test_scatter.png` — predicted vs actual
- `predicted_vs_actual.png`, `residuals_vs_predicted.png` — residual diagnostics
- `top_features.png` — top 25 descriptors by importance
- `pareto_ranked_screening.png` — Pareto screening plot

## Computational Notes

- Runtime is approximately 5–10 minutes on a modern laptop (CPU only; no GPU required)
- All random seeds are fixed (`RANDOM_STATE = 42`) for reproducibility
- The MLP is a 2-layer network trained with PyTorch; all other models use scikit-learn
