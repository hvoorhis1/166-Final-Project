import argparse
import itertools
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:
    raise ImportError(
        "PyTorch is required for this script. Install it in the active environment, e.g. `pip install torch`."
    ) from exc

plt.style.use("seaborn-v0_8-whitegrid")

RANDOM_STATE = 42
TARGET_COL = "deltaE_V"
NAME_COL = "Name"


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for a metric on the test set."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:
            continue
    lo = float(np.percentile(scores, (1 - ci) / 2 * 100))
    hi = float(np.percentile(scores, (1 + ci) / 2 * 100))
    return float(np.mean(scores)), lo, hi


@dataclass
class FeatureSet:
    name: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    all_df: pd.DataFrame


@dataclass
class Preprocessor:
    medians: pd.Series
    means: pd.Series
    stds: pd.Series

    @staticmethod
    def fit(df: pd.DataFrame, scale: bool = True) -> "Preprocessor":
        """Fit medians + z-score stats on train data."""
        med = df.median(numeric_only=True)
        x = df.fillna(med)
        if scale:
            means = x.mean()
            stds = x.std().replace(0, 1.0)
        else:
            means = pd.Series(0.0, index=x.columns)
            stds = pd.Series(1.0, index=x.columns)
        return Preprocessor(medians=med, means=means, stds=stds)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply imputation and scaling."""
        x = df.fillna(self.medians)
        x = (x - self.means) / self.stds
        return x.astype(np.float32).to_numpy()


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    project_dir = base_dir / "project"
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    plots_dir = project_dir / "plots"
    for p in [project_dir, data_dir, results_dir, plots_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {"project": project_dir, "data": data_dir, "results": results_dir, "plots": plots_dir}


def copy_datasets(source_dir: Path, data_dir: Path) -> dict[str, Path]:
    expected = [
        "DatasetQuinonesFiltered.csv",
        "DatasetQuinonesFilteredTrain.csv",
        "DatasetQuinonesFilteredTest.csv",
    ]
    copied: dict[str, Path] = {}
    for name in expected:
        src = source_dir / name
        if src.exists():
            dst = data_dir / name
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                dst.write_bytes(src.read_bytes())
            copied[name] = dst
    return copied


def get_train_test(data_paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_full = pd.read_csv(data_paths["DatasetQuinonesFiltered.csv"])
    if (
        "DatasetQuinonesFilteredTrain.csv" in data_paths
        and "DatasetQuinonesFilteredTest.csv" in data_paths
    ):
        return (
            df_full,
            pd.read_csv(data_paths["DatasetQuinonesFilteredTrain.csv"]),
            pd.read_csv(data_paths["DatasetQuinonesFilteredTest.csv"]),
        )

    y_bins = pd.qcut(df_full[TARGET_COL], q=5, labels=False, duplicates="drop")
    tr, ts = train_test_split(
        df_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bins
    )
    return df_full, tr.reset_index(drop=True), ts.reset_index(drop=True)


def print_eda(df: pd.DataFrame, plots_dir: Path) -> None:
    print("\\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print("\\nDtypes:")
    print(df.dtypes)
    print("\\nColumn names:")
    print(df.columns.tolist())
    print("\\nFirst 10 rows:")
    print(df.head(10))

    y = df[TARGET_COL]
    print("\\n=== TARGET SUMMARY (deltaE_V) ===")
    for k, v in {
        "mean": y.mean(),
        "std": y.std(),
        "min": y.min(),
        "25%": y.quantile(0.25),
        "median": y.median(),
        "75%": y.quantile(0.75),
        "max": y.max(),
    }.items():
        print(f"{k}: {v:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=30, alpha=0.8, edgecolor="black")
    ax.axvline(y.mean(), color="tab:red", linestyle="--", linewidth=2, label="Mean")
    ax.axvline(y.median(), color="tab:blue", linestyle="-.", linewidth=2, label="Median")
    ax.set_title("Distribution of deltaE_V")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "target_distribution.png", dpi=300)
    plt.close(fig)

    print("\\n=== MISSING VALUES PER COLUMN ===")
    print(df.isna().sum().sort_values(ascending=False))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    variances = df[numeric_cols].var(numeric_only=True)
    print(f"\\nConstant numeric columns ({(variances == 0).sum()}):")
    print(variances[variances == 0].index.tolist())
    print(f"Duplicated rows by {NAME_COL}: {df.duplicated(subset=[NAME_COL]).sum()}")

    desc_cols = [c for c in df.columns if c not in [NAME_COL, TARGET_COL]]
    non_num = [c for c in desc_cols if not pd.api.types.is_numeric_dtype(df[c])]
    print(f"Non-numeric descriptor columns ({len(non_num)}): {non_num}")

    num_desc = [c for c in desc_cols if c not in non_num]
    corr = df[num_desc].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().reset_index().rename(columns={"level_0": "f1", "level_1": "f2", 0: "corr"})
    pairs = pairs[pairs["corr"] > 0.95].sort_values("corr", ascending=False)
    print(f"High-correlation descriptor pairs (|r|>0.95): {len(pairs)}")


def choose_numeric_features(df_tr: pd.DataFrame, df_tst: pd.DataFrame) -> list[str]:
    candidates = [c for c in df_tr.columns if c not in [NAME_COL, TARGET_COL]]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df_tr[c])]

    xtr = df_tr[numeric].copy()
    all_missing = xtr.columns[xtr.isna().all()].tolist()
    variance = xtr.var(numeric_only=True)
    zero_var = variance[variance == 0].index.tolist()

    cleaned = [c for c in numeric if c not in set(all_missing + zero_var)]

    corr = xtr[cleaned].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
    cleaned = [c for c in cleaned if c not in drop_corr]

    missing_in_test = [c for c in cleaned if c not in df_tst.columns]
    if missing_in_test:
        raise ValueError(f"Missing cleaned features in test set: {missing_in_test}")

    print(f"Usable numeric descriptor columns after cleaning: {len(cleaned)}")
    return cleaned


def build_feature_sets(
    df_full: pd.DataFrame,
    df_tr: pd.DataFrame,
    df_tst: pd.DataFrame,
    full_features: list[str],
) -> tuple[list[FeatureSet], pd.DataFrame]:
    xtr = df_tr[full_features].copy()
    ytr = df_tr[TARGET_COL]

    corr_to_target = xtr.apply(lambda s: s.fillna(s.median()).corr(ytr), axis=0).abs().fillna(0)
    reduced_features = corr_to_target.sort_values(ascending=False).head(min(60, len(full_features))).index.tolist()

    impute_log = []
    for c in full_features:
        tr_miss = int(df_tr[c].isna().sum())
        ts_miss = int(df_tst[c].isna().sum())
        if tr_miss or ts_miss:
            impute_log.append({"column": c, "missing_train": tr_miss, "missing_test": ts_miss, "strategy": "median"})

    feature_sets = [
        FeatureSet("X_full", df_tr[full_features].copy(), df_tst[full_features].copy(), df_full[full_features].copy()),
        FeatureSet("X_reduced", df_tr[reduced_features].copy(), df_tst[reduced_features].copy(), df_full[reduced_features].copy()),
    ]
    for fs in feature_sets:
        print(f"{fs.name} dimensionality: train={fs.train_df.shape}, test={fs.test_df.shape}")
    return feature_sets, pd.DataFrame(impute_log)


def available_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def get_physics_sets(df_tr: pd.DataFrame) -> dict[str, list[str]]:
    a = available_cols(df_tr, ["LUMO"])
    b = available_cols(df_tr, ["LUMO", "HOMO"])
    c = available_cols(df_tr, ["LUMO", "HOMO", "dipole_moment", "meanI", "MAXDN2", "MAXDN", "minwHBa"])
    out = {}
    if a:
        out["Phys_A_LUMO"] = a
    if len(b) >= 2:
        out["Phys_B_LUMO_HOMO"] = b
    if len(c) >= 4:
        out["Phys_C_small_electronic"] = c
    return out


def get_descriptor_families(df_tr: pd.DataFrame) -> dict[str, list[str]]:
    family_defs = {
        "frontier_electronic": ["LUMO", "HOMO", "HOMO1", "HOMO2", "LUMO1", "dipole_moment", "P_alpha", "P_beta", "P_gamma"],
        "hbond_acceptor": ["nHBAcc", "nHBAcc2", "nHBAcc_Lipinski", "minwHBa", "maxwHBa", "MLFER_BH"],
        "topology_shape": ["nRing", "nFRing", "topoDiameter", "topoRadius", "topoShape", "Kier1", "Kier2", "Kier3", "ETA_Shape_P", "ETA_Shape_Y"],
        "charge_estate": ["MAXDN", "MAXDN2", "DELS", "DELS2", "GGI1", "GGI2", "GGI3", "JGI1", "JGI2", "JGI3", "SaasC", "maxaasC"],
        "size_mass_polarity": ["MW", "AMW", "TopoPSA", "VABC", "LipoaffinityIndex", "nAtom", "nHeavyAtom"],
    }
    return {k: available_cols(df_tr, v) for k, v in family_defs.items() if available_cols(df_tr, v)}


def make_torch_mlp_small(input_dim: int, hidden_dim: int = 64, dropout: float = 0.0) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def train_torch_mlp_small(x_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any]) -> nn.Module:
    set_seed()
    model = make_torch_mlp_small(
        input_dim=x_train.shape[1],
        hidden_dim=int(params.get("hidden_dim", 64)),
        dropout=float(params.get("dropout", 0.0)),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss()

    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1)),
    )
    loader = DataLoader(ds, batch_size=int(params.get("batch_size", 64)), shuffle=True)

    model.train()
    for _ in range(int(params.get("epochs", 300))):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model


def train_sklearn_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any]):
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=float(params["alpha"]), random_state=RANDOM_STATE)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown sklearn model: {model_name}")
    model.fit(x_train, y_train)
    return model


def predict_model(model_name: str, model: Any, x: np.ndarray) -> np.ndarray:
    if model_name == "TorchMLPSmall":
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(x)).squeeze(1).cpu().numpy()
    return model.predict(x)


def grid_dicts(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def get_model_specs() -> dict[str, dict[str, Any]]:
    return {
        "LinearRegression": {
            "scale": True,
            "grid": {"dummy": [0]},
            "cv_x": "dummy",
        },
        "Ridge": {
            "scale": True,
            "grid": {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]},
            "cv_x": "alpha",
        },
        "RandomForest": {
            "scale": False,
            "grid": {
                "n_estimators": [100, 300],
                "max_depth": [5, 10],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt"],
            },
            "cv_x": "max_depth",
        },
        "TorchMLPSmall": {
            "scale": True,
            "grid": {
                "hidden_dim": [64, 128],
                "dropout": [0.0, 0.1],
                "lr": [1e-3],
                "weight_decay": [1e-5, 1e-4],
                "epochs": [350],
                "batch_size": [64],
            },
            "cv_x": "hidden_dim",
        },
    }


def fit_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any]):
    if model_name == "TorchMLPSmall":
        return train_torch_mlp_small(x_train, y_train, params)
    return train_sklearn_model(model_name, x_train, y_train, params)


def cv_for_params(
    model_name: str,
    x_df: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    scale: bool,
) -> dict[str, float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmses, maes, r2s = [], [], []

    for tr_idx, va_idx in kf.split(x_df):
        x_tr_df, x_va_df = x_df.iloc[tr_idx], x_df.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx].to_numpy(), y.iloc[va_idx].to_numpy()

        prep = Preprocessor.fit(x_tr_df, scale=scale)
        x_tr = prep.transform(x_tr_df)
        x_va = prep.transform(x_va_df)

        model = fit_model(model_name, x_tr, y_tr, params)
        y_hat = predict_model(model_name, model, x_va)

        rmses.append(np.sqrt(mean_squared_error(y_va, y_hat)))
        maes.append(mean_absolute_error(y_va, y_hat))
        r2s.append(r2_score(y_va, y_hat))

    return {
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "cv_r2_mean": float(np.mean(r2s)),
        "cv_r2_std": float(np.std(r2s)),
    }


def evaluate_experiment(
    model_name: str,
    feature_set_name: str,
    x_tr_df: pd.DataFrame,
    x_tst_df: pd.DataFrame,
    x_all_df: pd.DataFrame,
    y_tr: pd.Series,
    y_tst: pd.Series,
    specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Grid search + CV, retrain best params, evaluate on test set."""
    cfg = specs[model_name]
    candidates = grid_dicts(cfg["grid"])
    cv_records = []
    best_score = -np.inf
    best_params = None
    best_cv = None

    for p in candidates:
        metrics = cv_for_params(model_name, x_tr_df, y_tr, p, cfg["scale"])
        cv_records.append({"params": p, **metrics})
        if metrics["cv_r2_mean"] > best_score:
            best_score = metrics["cv_r2_mean"]
            best_params = p
            best_cv = metrics

    assert best_params is not None and best_cv is not None

    prep = Preprocessor.fit(x_tr_df, scale=cfg["scale"])
    xtr = prep.transform(x_tr_df)
    xts = prep.transform(x_tst_df)

    model = fit_model(model_name, xtr, y_tr.to_numpy(), best_params)
    y_pred = predict_model(model_name, model, xts)

    high_cut = y_tst.quantile(0.8)
    high_mask = y_tst >= high_cut
    y_tst_high = y_tst[high_mask]
    y_pred_high = y_pred[high_mask.to_numpy()]

    row = {
        "model": model_name,
        "feature_set": feature_set_name,
        "best_params": json.dumps(best_params),
        **best_cv,
        "test_rmse": float(np.sqrt(mean_squared_error(y_tst, y_pred))),
        "test_mae": float(mean_absolute_error(y_tst, y_pred)),
        "test_r2": float(r2_score(y_tst, y_pred)),
        "high_rmse": float(np.sqrt(mean_squared_error(y_tst_high, y_pred_high))),
        "high_mae": float(mean_absolute_error(y_tst_high, y_pred_high)),
        "high_bias": float(np.mean(y_pred_high - y_tst_high.to_numpy())),
    }

    obj = {
        "model_name": model_name,
        "feature_set": feature_set_name,
        "preprocessor": prep,
        "model": model,
        "cv_records": cv_records,
        "cv_x": cfg["cv_x"],
        "x_all_df": x_all_df,
        "y_pred_test": y_pred,
    }
    return row, obj


def plot_cv_curve(cv_records: list[dict[str, Any]], x_key: str, title: str, out_path: Path) -> None:
    df = pd.DataFrame(cv_records)
    grp = df.groupby(df["params"].apply(lambda p: p.get(x_key, 0)))["cv_r2_mean"].max().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grp.index, grp.values, marker="o")
    if x_key in ["alpha", "weight_decay"]:
        ax.set_xscale("log")
    ax.set_xlabel(x_key)
    ax.set_ylabel("Mean CV R2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_test_scatter(y_true: pd.Series, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.8, label="Predictions")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "--", color="tab:red", linewidth=2, label="Ideal y=x")
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    xs = np.linspace(min_v, max_v, 100)
    ax.plot(xs, p(xs), color="tab:green", linewidth=2, label="Fitted trend")
    ax.set_xlabel("Actual deltaE_V")
    ax.set_ylabel("Predicted deltaE_V")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_model_comparison(results_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    pivot = results_df.pivot_table(index="model", columns="feature_set", values=metric)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison: {metric}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_top_features(feature_df: pd.DataFrame, out_path: Path, top_n: int = 25) -> None:
    if feature_df.empty:
        return
    value_col = None
    title = "Top Descriptor Importance"
    if "coef_abs" in feature_df.columns:
        value_col = "coef_abs"
        title = "Top Descriptor Importance (|Ridge Coefficients|)"
    elif "rf_importance" in feature_df.columns:
        value_col = "rf_importance"
        title = "Top Descriptor Importance (RandomForest)"
    elif "corr_proxy" in feature_df.columns:
        value_col = "corr_proxy"
        title = "Top Descriptor Importance (Correlation Proxy)"
    if value_col is None:
        return
    top = feature_df.sort_values(value_col, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"][::-1], top[value_col][::-1], color="tab:blue")
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def residual_analysis(names: pd.Series, y_true: pd.Series, y_pred: np.ndarray, plots_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    residuals = y_true - y_pred
    plot_test_scatter(y_true, y_pred, "Predicted vs Actual", plots_dir / "predicted_vs_actual.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.8)
    ax.axhline(0, color="tab:red", linestyle="--")
    ax.set_xlabel("Predicted deltaE_V")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_vs_predicted.png", dpi=300)
    plt.close(fig)

    stats_dict = {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_skewness": float(stats.skew(residuals)),
        "residual_kurtosis": float(stats.kurtosis(residuals)),
    }

    err_df = pd.DataFrame(
        {
            NAME_COL: names.values,
            "actual_deltaE_V": y_true.values,
            "predicted_deltaE_V": y_pred,
            "residual": residuals,
            "absolute_error": np.abs(residuals),
        }
    ).sort_values("absolute_error", ascending=False)
    return err_df, stats_dict


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    arr = df[[x_col, y_col]].to_numpy()
    is_pareto = np.ones(arr.shape[0], dtype=bool)
    for i, point in enumerate(arr):
        if not is_pareto[i]:
            continue
        dominates = np.all(arr >= point, axis=1) & np.any(arr > point, axis=1)
        if dominates.any():
            is_pareto[i] = False
    return pd.Series(is_pareto, index=df.index)


def screening_pipeline(
    df_full: pd.DataFrame,
    fs: FeatureSet,
    prep: Preprocessor,
    model: Any,
    model_name: str,
    y_train: pd.Series,
    results_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    """Rank all molecules by redox potential + capacity, apply filters, find Pareto front."""
    x_all = prep.transform(fs.all_df)
    preds = predict_model(model_name, model, x_all)
    out = pd.DataFrame({NAME_COL: df_full[NAME_COL], "predicted_redox_potential": preds})

    for c in ["MW", "TopoPSA", "LipoaffinityIndex", "dipole_moment", "nRing", "VABC", "LUMO", "HOMO", "nAtom", "nHeavyAtom", "topoDiameter"]:
        out[c] = df_full[c] if c in df_full.columns else np.nan

    F_CONST = 96485.33212
    out["capacity_proxy_mAh_g"] = 2 * F_CONST / (3.6 * out["MW"])

    # Weighted score: 50% potential, 35% capacity, 15% outlier penalty
    norm_pred = (out["predicted_redox_potential"] - out["predicted_redox_potential"].min()) / (
        out["predicted_redox_potential"].max() - out["predicted_redox_potential"].min() + 1e-12
    )
    norm_cap = (out["capacity_proxy_mAh_g"] - out["capacity_proxy_mAh_g"].min()) / (
        out["capacity_proxy_mAh_g"].max() - out["capacity_proxy_mAh_g"].min() + 1e-12
    )

    penalty = pd.Series(0.0, index=out.index)
    for c in ["MW", "LipoaffinityIndex", "TopoPSA", "nAtom", "nHeavyAtom", "VABC", "nRing", "topoDiameter", "LUMO", "HOMO"]:
        if out[c].notna().any():
            z = (out[c] - out[c].mean()) / (out[c].std() + 1e-12)
            penalty += np.clip(np.abs(z) - 2.0, 0, None)
    penalty = penalty / (penalty.max() + 1e-12)

    out["screen_score"] = 0.50 * norm_pred + 0.35 * norm_cap - 0.15 * penalty
    out["screen_rank"] = out["screen_score"].rank(ascending=False, method="min").astype(int)

    # Hard filters for practical viability
    out["passes_mw"] = out["MW"] <= 500
    pol = pd.Series(True, index=out.index)
    for c in ["TopoPSA", "LipoaffinityIndex", "dipole_moment"]:
        if out[c].notna().any():
            lo, hi = out[c].quantile([0.05, 0.95])
            pol &= out[c].between(lo, hi)
    out["passes_polarity"] = pol

    comp = pd.Series(True, index=out.index)
    for c in ["nAtom", "nHeavyAtom", "VABC", "nRing", "topoDiameter"]:
        if out[c].notna().any():
            comp &= out[c] <= out[c].quantile(0.95)
    out["passes_complexity"] = comp

    elec = pd.Series(True, index=out.index)
    for c in ["LUMO", "HOMO"]:
        if out[c].notna().any():
            lo, hi = out[c].quantile([0.01, 0.99])
            elec &= out[c].between(lo, hi)
    p_lo, p_hi = np.quantile(y_train, [0.01, 0.99])
    elec &= out["predicted_redox_potential"].between(p_lo, p_hi)
    out["passes_electronic"] = elec
    out["passes_all"] = out["passes_mw"] & out["passes_polarity"] & out["passes_complexity"] & out["passes_electronic"]

    out["pareto_frontier"] = pareto_frontier(out, "capacity_proxy_mAh_g", "predicted_redox_potential")

    out.to_csv(results_dir / "screening_results.csv", index=False)

    top20 = out.sort_values("screen_score", ascending=False).head(20).copy()
    top20.to_csv(results_dir / "top20_candidates.csv", index=False)

    # Screening plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(out["capacity_proxy_mAh_g"], out["predicted_redox_potential"], c="lightgray", alpha=0.6, label="All")
    top_idx = out.sort_values("screen_score", ascending=False).head(50).index
    ax.scatter(
        out.loc[top_idx, "capacity_proxy_mAh_g"],
        out.loc[top_idx, "predicted_redox_potential"],
        c="tab:blue",
        alpha=0.8,
        label="Top 50 by score",
    )
    pf = out[out["pareto_frontier"]]
    ax.scatter(pf["capacity_proxy_mAh_g"], pf["predicted_redox_potential"], c="tab:red", alpha=0.9, label="Pareto")
    ax.set_xlabel("Capacity proxy (mAh/g)")
    ax.set_ylabel("Predicted redox potential")
    ax.set_title("Screening: Pareto + Ranked Candidates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "pareto_ranked_screening.png", dpi=300)
    plt.close(fig)

    print("\\n=== SCREENING SUMMARY ===")
    print(f"Top 10 by score: {len(out.sort_values('screen_score', ascending=False).head(10))}")
    print(f"Top 20 by score: {len(out.sort_values('screen_score', ascending=False).head(20))}")
    print(f"Top 50 by score: {len(out.sort_values('screen_score', ascending=False).head(50))}")
    print(f"Top 5% by score count: {int(np.ceil(0.05 * len(out)))}")

    print("\\n=== TOP 20 CANDIDATES (by screen_score) ===")
    print(
        top20[
            [
                NAME_COL,
                "screen_score",
                "predicted_redox_potential",
                "capacity_proxy_mAh_g",
                "MW",
                "TopoPSA",
                "LipoaffinityIndex",
                "dipole_moment",
                "nRing",
                "VABC",
                "LUMO",
                "HOMO",
                "passes_all",
                "pareto_frontier",
            ]
        ]
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redox potential prediction and screening pipeline for quinone molecules."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional path to external data directory containing dataset CSVs. "
             "If omitted, data is loaded from project/data/.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    set_seed()

    base_dir = Path(__file__).resolve().parents[1]
    dirs = ensure_dirs(base_dir)
    for png in dirs["plots"].glob("*.png"):
        png.unlink()

    # Load data
    if args.data_dir is not None:
        data_paths = copy_datasets(Path(args.data_dir), dirs["data"])
    else:
        data_paths = {}
        for name in [
            "DatasetQuinonesFiltered.csv",
            "DatasetQuinonesFilteredTrain.csv",
            "DatasetQuinonesFilteredTest.csv",
        ]:
            p = dirs["data"] / name
            if p.exists():
                data_paths[name] = p

    if "DatasetQuinonesFiltered.csv" not in data_paths:
        raise FileNotFoundError(
            "DatasetQuinonesFiltered.csv not found in project/data/. "
            "See README.md for data acquisition instructions."
        )

    df_full, df_tr, df_tst = get_train_test(data_paths)
    print_eda(df_full, dirs["plots"])

    full_features = choose_numeric_features(df_tr, df_tst)
    feature_sets, impute_log = build_feature_sets(df_full, df_tr, df_tst, full_features)
    impute_log.to_csv(dirs["data"] / "preprocessing_log.csv", index=False)

    y_tr = df_tr[TARGET_COL]
    y_tst = df_tst[TARGET_COL]

    # Experiments
    specs = get_model_specs()
    fs_map = {fs.name: fs for fs in feature_sets}

    experiments: list[tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

    # Physics baselines
    phys_sets = get_physics_sets(df_tr)
    for pname, cols in phys_sets.items():
        experiments.append(("LinearRegression", pname, df_tr[cols], df_tst[cols], df_full[cols]))
        experiments.append(("Ridge", pname, df_tr[cols], df_tst[cols], df_full[cols]))

    # Full descriptor models
    x_full = fs_map["X_full"]
    experiments.extend(
        [
            ("Ridge", "X_full", x_full.train_df, x_full.test_df, x_full.all_df),
            ("RandomForest", "X_full", x_full.train_df, x_full.test_df, x_full.all_df),
            ("TorchMLPSmall", "X_full", x_full.train_df, x_full.test_df, x_full.all_df),
        ]
    )

    # Reduced feature set
    x_red = fs_map["X_reduced"]
    experiments.extend(
        [
            ("Ridge", "X_reduced", x_red.train_df, x_red.test_df, x_red.all_df),
            ("RandomForest", "X_reduced", x_red.train_df, x_red.test_df, x_red.all_df),
            ("TorchMLPSmall", "X_reduced", x_red.train_df, x_red.test_df, x_red.all_df),
        ]
    )

    rows = []
    objects: dict[str, dict[str, Any]] = {}
    for model_name, fs_name, xtr_df, xts_df, xall_df in experiments:
        row, obj = evaluate_experiment(model_name, fs_name, xtr_df, xts_df, xall_df, y_tr, y_tst, specs)
        rows.append(row)
        key = f"{model_name}::{fs_name}"
        objects[key] = obj
        print(
            f"{model_name} on {fs_name}: test RMSE={row['test_rmse']:.4f}, "
            f"MAE={row['test_mae']:.4f}, R2={row['test_r2']:.4f}"
        )

    results_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    results_df.to_csv(dirs["results"] / "model_comparison.csv", index=False)

    # Descriptor family ablation
    families = get_descriptor_families(df_tr)
    fam_rows = []
    for fam_name, cols in families.items():
        row, _ = evaluate_experiment(
            "Ridge",
            f"family_{fam_name}",
            df_tr[cols],
            df_tst[cols],
            df_full[cols],
            y_tr,
            y_tst,
            specs,
        )
        fam_rows.append({"family": fam_name, "test_r2": row["test_r2"], "test_rmse": row["test_rmse"]})
    family_df = pd.DataFrame(fam_rows).sort_values("test_r2", ascending=False)
    family_df.to_csv(dirs["results"] / "descriptor_family_performance.csv", index=False)

    # Family performance plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(family_df["family"], family_df["test_r2"], color="tab:blue")
    ax.set_ylabel("Test R2")
    ax.set_title("Descriptor Family Performance (Ridge)")
    ax.set_xticklabels(family_df["family"], rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "descriptor_family_performance.png", dpi=300)
    plt.close(fig)

    # Physics vs ML plot (use short readable labels)
    mask_phys = results_df["feature_set"].str.startswith("Phys_")
    mask_full = (results_df["feature_set"] == "X_full") & (results_df["model"].isin(["Ridge", "RandomForest", "TorchMLPSmall"]))
    phys_ml_df = results_df[mask_phys | mask_full].copy()
    short_fs = {"Phys_A_LUMO": "LUMO", "Phys_B_LUMO_HOMO": "LUMO+HOMO", "Phys_C_small_electronic": "Phys-C (7)", "X_full": "Full (141)"}
    short_model = {"LinearRegression": "LinReg", "Ridge": "Ridge", "RandomForest": "RF", "TorchMLPSmall": "MLP"}
    phys_ml_df["label"] = phys_ml_df.apply(lambda r: f"{short_fs.get(r['feature_set'], r['feature_set'])}\n{short_model.get(r['model'], r['model'])}", axis=1)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(phys_ml_df)), phys_ml_df["test_r2"], color="tab:green")
    ax.set_xticks(range(len(phys_ml_df)))
    ax.set_xticklabels(phys_ml_df["label"], fontsize=10)
    ax.set_ylabel("Test $R^2$", fontsize=12)
    ax.set_title("Physics Baselines vs Full ML Models", fontsize=13)
    for bar, val in zip(bars, phys_ml_df["test_r2"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "physics_vs_ml_r2.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(phys_ml_df)), phys_ml_df["test_rmse"], color="tab:orange")
    ax.set_xticks(range(len(phys_ml_df)))
    ax.set_xticklabels(phys_ml_df["label"], fontsize=10)
    ax.set_ylabel("Test RMSE (V)", fontsize=12)
    ax.set_title("Physics Baselines vs Full ML Models", fontsize=13)
    for bar, val in zip(bars, phys_ml_df["test_rmse"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "physics_vs_ml_rmse.png", dpi=300)
    plt.close(fig)

    # CV curves and scatter plots
    cv_name_map = {
        "Ridge": "cv_ridge.png",
        "RandomForest": "cv_random_forest.png",
        "TorchMLPSmall": "cv_torch_mlp_small.png",
    }
    scatter_name_map = {
        "Ridge": "ridge_test_scatter.png",
        "RandomForest": "random_forest_test_scatter.png",
        "TorchMLPSmall": "torch_mlp_small_test_scatter.png",
    }

    for model_name, out_name in cv_name_map.items():
        sub = results_df[(results_df["model"] == model_name) & (results_df["feature_set"] == "X_full")]
        if sub.empty:
            continue
        key = f"{model_name}::X_full"
        obj = objects[key]
        plot_cv_curve(obj["cv_records"], obj["cv_x"], f"CV Tuning Curve: {model_name}", dirs["plots"] / out_name)

    for model_name, out_name in scatter_name_map.items():
        sub = results_df[(results_df["model"] == model_name) & (results_df["feature_set"] == "X_full")]
        if sub.empty:
            continue
        key = f"{model_name}::X_full"
        obj = objects[key]
        plot_test_scatter(y_tst, obj["y_pred_test"], f"{model_name} Predicted vs Actual", dirs["plots"] / out_name)

    # LUMO-only baseline scatter
    if "Ridge::Phys_A_LUMO" in objects:
        obj = objects["Ridge::Phys_A_LUMO"]
        plot_test_scatter(y_tst, obj["y_pred_test"], "Ridge on LUMO-only", dirs["plots"] / "baseline_lumo_scatter.png")

    plot_model_comparison(results_df, "test_r2", dirs["plots"] / "model_comparison_r2.png")
    plot_model_comparison(results_df, "test_rmse", dirs["plots"] / "model_comparison_rmse.png")

    # High-potential regime plot
    hp = results_df[(results_df["feature_set"].isin(["Phys_A_LUMO", "X_full"])) & (results_df["model"].isin(["Ridge", "RandomForest", "TorchMLPSmall", "LinearRegression"]))].copy()
    hp["label"] = hp["feature_set"] + "|" + hp["model"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(hp["label"], hp["high_rmse"], color="tab:purple")
    ax.set_ylabel("High-Potential RMSE")
    ax.set_title("High-Potential Regime Performance (Top 20% deltaE_V)")
    ax.set_xticklabels(hp["label"], rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "high_potential_performance.png", dpi=300)
    plt.close(fig)

    # Pick best model for screening
    best_full = results_df[(results_df["feature_set"] == "X_full") & (results_df["model"].isin(["Ridge", "RandomForest", "TorchMLPSmall"]))].sort_values("test_r2", ascending=False).iloc[0]
    best_key = f"{best_full['model']}::{best_full['feature_set']}"
    best_obj = objects[best_key]

    # Feature importance
    x_full_cols = fs_map["X_full"].train_df.columns
    if best_full["model"] == "Ridge":
        coef_abs = np.abs(best_obj["model"].coef_)
        feat_rank = pd.DataFrame({"feature": x_full_cols, "coef_abs": coef_abs}).sort_values("coef_abs", ascending=False)
        feat_rank["coef_rank"] = np.arange(1, len(feat_rank) + 1)
    elif best_full["model"] == "RandomForest":
        imp = best_obj["model"].feature_importances_
        feat_rank = pd.DataFrame({"feature": x_full_cols, "rf_importance": imp}).sort_values("rf_importance", ascending=False)
        feat_rank["rf_rank"] = np.arange(1, len(feat_rank) + 1)
    else:
        corr_proxy = fs_map["X_full"].train_df.apply(lambda s: s.fillna(s.median()).corr(y_tr), axis=0).abs().fillna(0)
        feat_rank = corr_proxy.sort_values(ascending=False).reset_index()
        feat_rank.columns = ["feature", "corr_proxy"]
        feat_rank["proxy_rank"] = np.arange(1, len(feat_rank) + 1)

    feat_rank.to_csv(dirs["results"] / "feature_ranking.csv", index=False)
    plot_top_features(feat_rank, dirs["plots"] / "top_features.png", top_n=25)

    # Residuals
    err_df, residual_stats = residual_analysis(df_tst[NAME_COL], y_tst, best_obj["y_pred_test"], dirs["plots"])
    print("\\n=== RESIDUAL STATS ===")
    print(residual_stats)
    print("\\n10 worst predictions:")
    print(err_df.head(10))

    # Substituent / structural-type error analysis
    y_pred_best = best_obj["y_pred_test"]
    abs_err = np.abs(y_tst.to_numpy() - y_pred_best)
    sub_df = df_tst[[NAME_COL]].copy()
    sub_df["abs_error"] = abs_err
    sub_df["actual"] = y_tst.values
    sub_df["predicted"] = y_pred_best
    # classify by structural features available in the test set
    sub_df["has_nitrogen"] = (df_tst["nN"] > 0).values if "nN" in df_tst.columns else False
    sub_df["is_aromatic"] = (df_tst["naAromAtom"] > 0).values if "naAromAtom" in df_tst.columns else False
    sub_df["n_rings"] = df_tst["nRing"].values if "nRing" in df_tst.columns else 0
    sub_df["ring_group"] = pd.cut(sub_df["n_rings"], bins=[-1, 1, 2, 3, 100], labels=["1", "2", "3", "4+"])
    sub_df["is_variant"] = sub_df[NAME_COL].str.contains("_prop_|_\\d+$", regex=True)
    sub_df["mw_group"] = pd.cut(df_tst["MW"].values, bins=[0, 150, 200, 250, 1000], labels=["<150", "150-200", "200-250", ">250"])

    print("\n=== ERROR BY STRUCTURAL TYPE ===")
    for col in ["has_nitrogen", "is_aromatic", "ring_group", "is_variant", "mw_group"]:
        grp = sub_df.groupby(col)["abs_error"].agg(["mean", "std", "count"])
        print(f"\nBy {col}:")
        print(grp.round(4))

    # Plot error by ring count and MW group
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(axes, ["ring_group", "mw_group"], ["Ring Count", "Molecular Weight"]):
        grp = sub_df.groupby(col)["abs_error"].agg(["mean", "count"])
        ax.bar(grp.index.astype(str), grp["mean"], color="tab:blue")
        for i, (idx, row_g) in enumerate(grp.iterrows()):
            ax.text(i, row_g["mean"] + 0.002, f"n={int(row_g['count'])}", ha="center", fontsize=9)
        ax.set_xlabel(title)
        ax.set_ylabel("Mean Absolute Error (V)")
        ax.set_title(f"Prediction Error by {title}")
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "error_by_structure.png", dpi=300)
    plt.close(fig)

    sub_df.to_csv(dirs["results"] / "error_by_structure.csv", index=False)

    screening = screening_pipeline(
        df_full,
        fs_map["X_full"],
        best_obj["preprocessor"],
        best_obj["model"],
        str(best_full["model"]),
        y_tr,
        dirs["results"],
        dirs["plots"],
    )

    # Print results with bootstrap CIs
    print("\n=== RESULTS TABLE (with 95% bootstrap CI on test set) ===")
    header = f"{'Model':<20} {'Features':<16} {'Test R²':>22}  {'Test RMSE':>22}  {'Test MAE':>22}"
    print(header)
    print("-" * len(header))
    y_tst_np = y_tst.to_numpy()
    ci_rows = []
    for _, r in results_df.iterrows():
        key = f"{r['model']}::{r['feature_set']}"
        y_pred = objects[key]["y_pred_test"]
        r2_mean, r2_lo, r2_hi = bootstrap_ci(y_tst_np, y_pred, r2_score)
        rmse_mean, rmse_lo, rmse_hi = bootstrap_ci(
            y_tst_np, y_pred, lambda y, p: np.sqrt(mean_squared_error(y, p))
        )
        mae_mean, mae_lo, mae_hi = bootstrap_ci(
            y_tst_np, y_pred, mean_absolute_error
        )
        ci_rows.append({
            "model": r["model"], "feature_set": r["feature_set"],
            "test_r2": r2_mean, "r2_ci_lo": r2_lo, "r2_ci_hi": r2_hi,
            "test_rmse": rmse_mean, "rmse_ci_lo": rmse_lo, "rmse_ci_hi": rmse_hi,
            "test_mae": mae_mean, "mae_ci_lo": mae_lo, "mae_ci_hi": mae_hi,
        })
        print(
            f"{r['model']:<20} {r['feature_set']:<16} "
            f"{r2_mean:>6.4f} [{r2_lo:.4f}, {r2_hi:.4f}]  "
            f"{rmse_mean:>6.4f} [{rmse_lo:.4f}, {rmse_hi:.4f}]  "
            f"{mae_mean:>6.4f} [{mae_lo:.4f}, {mae_hi:.4f}]"
        )
    pd.DataFrame(ci_rows).to_csv(dirs["results"] / "results_with_ci.csv", index=False)

    print("\n=== FINAL SUMMARY ===")
    print(
        f"Best full model: {best_full['model']} on X_full | "
        f"Test RMSE={best_full['test_rmse']:.4f}, MAE={best_full['test_mae']:.4f}, R2={best_full['test_r2']:.4f}"
    )
    print(f"Molecules passing strict filters: {int(screening['passes_all'].sum())}")


if __name__ == "__main__":
    main()
