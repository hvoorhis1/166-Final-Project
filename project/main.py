import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

plt.style.use("seaborn-v0_8-whitegrid")

RANDOM_STATE = 42
N_JOBS = 1
TARGET_COL = "deltaE_V"
NAME_COL = "Name"



@dataclass
class FeatureSet:
    name: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    all_df: pd.DataFrame


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    project_dir = base_dir / "project"
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    plots_dir = project_dir / "plots"
    for p in [project_dir, data_dir, results_dir, plots_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {
        "project": project_dir,
        "data": data_dir,
        "results": results_dir,
        "plots": plots_dir,
    }


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


def print_eda(df: pd.DataFrame, plots_dir: Path) -> None:
    print("\\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print("\\nDtypes:")
    print(df.dtypes)
    print("\\nColumn names:")
    print(df.columns.tolist())
    print("\\nFirst 10 rows:")
    print(df.head(10))

    print("\\n=== TARGET SUMMARY (deltaE_V) ===")
    target = df[TARGET_COL]
    summary = {
        "mean": target.mean(),
        "std": target.std(),
        "min": target.min(),
        "25%": target.quantile(0.25),
        "median": target.median(),
        "75%": target.quantile(0.75),
        "max": target.max(),
    }
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(target, bins=30, alpha=0.8, edgecolor="black")
    ax.axvline(target.mean(), color="tab:red", linestyle="--", linewidth=2, label="Mean")
    ax.axvline(target.median(), color="tab:blue", linestyle="-.", linewidth=2, label="Median")
    ax.set_title("Distribution of deltaE_V")
    ax.set_xlabel("deltaE_V")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "target_distribution.png", dpi=300)
    plt.close(fig)

    print("\\n=== MISSING VALUES PER COLUMN ===")
    print(df.isna().sum().sort_values(ascending=False))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    variances = df[numeric_cols].var(numeric_only=True)
    constant_cols = variances[variances == 0].index.tolist()
    near_constant_cols = variances[variances < 1e-6].index.tolist()
    print(f"\\nConstant numeric columns ({len(constant_cols)}): {constant_cols}")
    print(f"Nearly constant numeric columns ({len(near_constant_cols)}): {near_constant_cols[:20]}")

    dup_name_count = df.duplicated(subset=[NAME_COL]).sum() if NAME_COL in df.columns else 0
    print(f"\\nDuplicated rows by {NAME_COL}: {dup_name_count}")

    descriptor_cols = [c for c in df.columns if c not in [NAME_COL, TARGET_COL]]
    non_numeric_descriptors = [
        c for c in descriptor_cols if not pd.api.types.is_numeric_dtype(df[c])
    ]
    print(
        f"Non-numeric descriptor columns ({len(non_numeric_descriptors)}): {non_numeric_descriptors}"
    )

    numeric_desc = [c for c in descriptor_cols if c not in non_numeric_descriptors]
    corr = df[numeric_desc].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_pairs = (
        upper.stack().reset_index().rename(columns={"level_0": "f1", "level_1": "f2", 0: "corr"})
    )
    high_corr_pairs = high_corr_pairs[high_corr_pairs["corr"] > 0.95].sort_values("corr", ascending=False)
    print(f"\\nHigh-correlation descriptor pairs (|r|>0.95): {len(high_corr_pairs)}")
    if not high_corr_pairs.empty:
        print(high_corr_pairs.head(20))


def get_train_test(data_paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_full = pd.read_csv(data_paths["DatasetQuinonesFiltered.csv"])
    if (
        "DatasetQuinonesFilteredTrain.csv" in data_paths
        and "DatasetQuinonesFilteredTest.csv" in data_paths
    ):
        df_tr = pd.read_csv(data_paths["DatasetQuinonesFilteredTrain.csv"])
        df_tst = pd.read_csv(data_paths["DatasetQuinonesFilteredTest.csv"])
        print("\\nUsing predefined train/test splits from dataset files.")
        return df_full, df_tr, df_tst

    print("\\nPredefined split not found; using stratified 80/20 split.")
    y_bins = pd.qcut(df_full[TARGET_COL], q=5, labels=False, duplicates="drop")
    df_tr, df_tst = train_test_split(
        df_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bins
    )
    return df_full, df_tr.reset_index(drop=True), df_tst.reset_index(drop=True)


def choose_numeric_features(df_tr: pd.DataFrame, df_tst: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [c for c in df_tr.columns if c not in [NAME_COL, TARGET_COL]]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df_tr[c])]
    dropped = [c for c in candidates if c not in numeric]

    train_numeric = df_tr[numeric].copy()

    all_missing = train_numeric.columns[train_numeric.isna().all()].tolist()
    variance = train_numeric.var(numeric_only=True)
    zero_var = variance[variance == 0].index.tolist()

    cleaned = [c for c in numeric if c not in set(all_missing + zero_var)]

    corr = train_numeric[cleaned].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
    cleaned = [c for c in cleaned if c not in to_drop_corr]

    print(f"\\nUsable numeric descriptor columns after cleaning: {len(cleaned)}")
    print(f"Dropped non-numeric: {len(dropped)}")
    print(f"Dropped all-missing: {len(all_missing)}")
    print(f"Dropped zero-variance: {len(zero_var)}")
    print(f"Dropped high-correlation (>|0.95|): {len(to_drop_corr)}")

    missing_in_test = [c for c in cleaned if c not in df_tst.columns]
    if missing_in_test:
        raise ValueError(f"Missing cleaned features in test set: {missing_in_test}")

    return cleaned, to_drop_corr


def build_feature_sets(
    df_full: pd.DataFrame,
    df_tr: pd.DataFrame,
    df_tst: pd.DataFrame,
    full_features: list[str],
) -> tuple[list[FeatureSet], pd.DataFrame]:
    impute_log_rows: list[dict[str, Any]] = []

    # Reduced features via LassoCV on standardized + imputed full feature set.
    reducer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoCV(
                    cv=5,
                    random_state=RANDOM_STATE,
                    max_iter=20000,
                    n_alphas=100,
                ),
            ),
        ]
    )
    reducer.fit(df_tr[full_features], df_tr[TARGET_COL])
    coefs = pd.Series(reducer.named_steps["lasso"].coef_, index=full_features)
    reduced_features = coefs[coefs.abs() > 1e-8].index.tolist()
    if len(reduced_features) < 10:
        reduced_features = coefs.abs().sort_values(ascending=False).head(25).index.tolist()
    print(f"Reduced feature set size: {len(reduced_features)}")

    for col in full_features:
        miss_tr = int(df_tr[col].isna().sum())
        miss_tst = int(df_tst[col].isna().sum())
        if miss_tr or miss_tst:
            impute_log_rows.append(
                {
                    "column": col,
                    "missing_train": miss_tr,
                    "missing_test": miss_tst,
                    "strategy": "median",
                }
            )

    feature_sets = [
        FeatureSet(
            name="X_full",
            train_df=df_tr[full_features].copy(),
            test_df=df_tst[full_features].copy(),
            all_df=df_full[full_features].copy(),
        ),
        FeatureSet(
            name="X_reduced",
            train_df=df_tr[reduced_features].copy(),
            test_df=df_tst[reduced_features].copy(),
            all_df=df_full[reduced_features].copy(),
        ),
    ]

    for fs in feature_sets:
        print(f"{fs.name} dimensionality: train={fs.train_df.shape}, test={fs.test_df.shape}")

    return feature_sets, pd.DataFrame(impute_log_rows)


def get_models_and_grids() -> dict[str, tuple[Any, dict[str, list[Any]], bool]]:
    return {
        "Ridge": (
            Ridge(random_state=RANDOM_STATE),
            {"model__alpha": np.logspace(-4, 1, 30).tolist()},
            True,
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            {
                "model__n_estimators": [50, 100, 300],
                "model__max_depth": [3, 5, 10],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", None],
            },
            False,
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [10, 30, 50, 100],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.5],
            },
            False,
        ),
        "Lasso": (
            LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000),
            {},
            True,
        ),
    }


def build_pipeline(estimator: Any, scale: bool) -> Pipeline:
    preprocessor_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        preprocessor_steps.append(("scaler", StandardScaler()))
    preprocessor = Pipeline(preprocessor_steps)

    return Pipeline(
        steps=[
            (
                "pre",
                ColumnTransformer(
                    transformers=[("num", preprocessor, slice(0, None))],
                    remainder="drop",
                ),
            ),
            ("model", estimator),
        ]
    )


def evaluate_feature_sets(
    feature_sets: list[FeatureSet],
    y_tr: pd.Series,
    y_tst: pd.Series,
    plots_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    models = get_models_and_grids()
    results = []
    best_objects: dict[str, dict[str, Any]] = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    for fs in feature_sets:
        X_tr = fs.train_df
        X_tst = fs.test_df

        for model_name, (estimator, param_grid, scale_flag) in models.items():
            pipe = build_pipeline(estimator, scale=scale_flag)

            if isinstance(estimator, LassoCV):
                cv_scores = cross_validate(
                    pipe,
                    X_tr,
                    y_tr,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=N_JOBS,
                )
                pipe.fit(X_tr, y_tr)
                best_est = pipe
                best_params = {
                    "alpha": float(
                        best_est.named_steps["model"].alpha_
                    ),
                    "nonzero_coef": int(
                        np.sum(np.abs(best_est.named_steps["model"].coef_) > 1e-8)
                    ),
                }
                cv_results = None
            else:
                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    scoring="r2",
                    cv=cv,
                    n_jobs=N_JOBS,
                    return_train_score=True,
                )
                gs.fit(X_tr, y_tr)
                best_est = gs.best_estimator_
                best_params = gs.best_params_
                cv_results = gs.cv_results_

                cv_scores = cross_validate(
                    clone(best_est),
                    X_tr,
                    y_tr,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=N_JOBS,
                )

            y_pred = best_est.predict(X_tst)
            test_rmse = np.sqrt(mean_squared_error(y_tst, y_pred))
            test_mae = mean_absolute_error(y_tst, y_pred)
            test_r2 = r2_score(y_tst, y_pred)

            row = {
                "model": model_name,
                "feature_set": fs.name,
                "best_params": json.dumps(best_params),
                "cv_rmse_mean": float(-cv_scores["test_rmse"].mean()),
                "cv_rmse_std": float(cv_scores["test_rmse"].std()),
                "cv_mae_mean": float(-cv_scores["test_mae"].mean()),
                "cv_mae_std": float(cv_scores["test_mae"].std()),
                "cv_r2_mean": float(cv_scores["test_r2"].mean()),
                "cv_r2_std": float(cv_scores["test_r2"].std()),
                "test_rmse": float(test_rmse),
                "test_mae": float(test_mae),
                "test_r2": float(test_r2),
            }
            results.append(row)

            key = f"{model_name}::{fs.name}"
            best_objects[key] = {
                "model_name": model_name,
                "feature_set": fs,
                "estimator": best_est,
                "best_params": best_params,
                "cv_results": cv_results,
                "y_pred_test": y_pred,
            }

            print(
                f"{model_name} on {fs.name}: test RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R2={test_r2:.4f}"
            )

    results_df = pd.DataFrame(results).sort_values(by="test_r2", ascending=False)
    return results_df, best_objects


def plot_cv_curve(cv_results: dict[str, Any], model_name: str, out_path: Path) -> None:
    if cv_results is None:
        return

    df = pd.DataFrame(cv_results)
    fig, ax = plt.subplots(figsize=(8, 5))

    if model_name == "Ridge":
        x = df["param_model__alpha"].astype(float)
        ax.plot(x, df["mean_test_score"], marker="o", label="CV R2")
        ax.plot(x, df["mean_train_score"], marker="x", label="Train R2")
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
    elif model_name == "RandomForest":
        group = df.groupby("param_model__max_depth", dropna=False)["mean_test_score"].max()
        x = [str(v) for v in group.index]
        ax.plot(x, group.values, marker="o", label="Best CV R2 by max_depth")
        ax.set_xlabel("max_depth")
    elif model_name == "GradientBoosting":
        for lr in sorted(df["param_model__learning_rate"].astype(float).unique()):
            sub = df[df["param_model__learning_rate"].astype(float) == lr]
            grp = sub.groupby("param_model__n_estimators")["mean_test_score"].max().sort_index()
            ax.plot(grp.index.astype(int), grp.values, marker="o", label=f"lr={lr}")
        ax.set_xlabel("n_estimators")

    ax.set_ylabel("Mean CV R2")
    ax.set_title(f"CV Tuning Curve: {model_name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_test_scatter(y_true: pd.Series, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.8, label="Predictions")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label="Ideal y=x",
    )
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
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison: {metric}")
    ax.legend(title="Feature Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def top_feature_plot(
    feature_names: list[str],
    values: np.ndarray,
    out_path: Path,
    title: str,
    color_by_sign: bool = False,
) -> pd.DataFrame:
    ser = pd.Series(values, index=feature_names)
    top = ser.abs().sort_values(ascending=False).head(25)
    plot_vals = ser[top.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    if color_by_sign:
        colors = ["tab:blue" if v >= 0 else "tab:red" for v in plot_vals.values]
    else:
        colors = "tab:blue"
    ax.barh(plot_vals.index[::-1], plot_vals.values[::-1], color=colors)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return plot_vals.to_frame("importance").reset_index().rename(columns={"index": "feature"})


def residual_analysis(
    names: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    plots_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    residuals = y_true - y_pred

    plot_test_scatter(y_true, y_pred, "Predicted vs Actual", plots_dir / "predicted_vs_actual.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.8)
    ax.axhline(0, color="tab:red", linestyle="--")
    ax.set_xlabel("Predicted deltaE_V")
    ax.set_ylabel("Residuals")
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

    y_bins = pd.qcut(y_true, q=5, duplicates="drop")
    by_bin = (
        pd.DataFrame({"bin": y_bins, "actual": y_true, "pred": y_pred})
        .groupby("bin")
        .apply(
            lambda g: pd.Series(
                {
                    "count": len(g),
                    "MAE": mean_absolute_error(g["actual"], g["pred"]),
                    "RMSE": np.sqrt(mean_squared_error(g["actual"], g["pred"])),
                }
            )
        )
        .reset_index()
    )

    return err_df, stats_dict, by_bin


def descriptor_class_analysis(
    df_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    plots_dir: Path,
) -> pd.DataFrame:
    residual_abs = np.abs(y_true - y_pred)
    df = df_test.copy()
    df["abs_err"] = residual_abs
    df["actual"] = y_true.values
    df["pred"] = y_pred

    classes: dict[str, pd.Series] = {}
    potential_cols = {
        "high_topopsa": "TopoPSA",
        "high_mw": "MW",
        "many_rings": "nRing",
        "high_dipole": "dipole_moment",
        "low_lumo": "LUMO",
        "high_lumo": "LUMO",
        "high_lipo": "LipoaffinityIndex",
        "many_heavy_atoms": "nHeavyAtom",
    }

    for cname, col in potential_cols.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if cname.startswith("low_"):
                thresh = df[col].quantile(0.2)
                classes[cname] = df[col] <= thresh
            else:
                thresh = df[col].quantile(0.8)
                classes[cname] = df[col] >= thresh

    rows = []
    for cname, mask in classes.items():
        sub = df[mask]
        if len(sub) < 5:
            continue
        mae = mean_absolute_error(sub["actual"], sub["pred"])
        rmse = np.sqrt(mean_squared_error(sub["actual"], sub["pred"]))
        rows.append(
            {
                "class": cname,
                "count": len(sub),
                "mean_target": sub["actual"].mean(),
                "MAE": mae,
                "RMSE": rmse,
            }
        )

    class_df = pd.DataFrame(rows).sort_values("MAE", ascending=False)
    return class_df


def screening_pipeline(
    df_full: pd.DataFrame,
    fs: FeatureSet,
    best_estimator: Pipeline,
    y_train: pd.Series,
    results_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    preds = best_estimator.predict(fs.all_df)
    out = pd.DataFrame({NAME_COL: df_full[NAME_COL], "predicted_redox_potential": preds})

    needed_cols = [
        "MW",
        "TopoPSA",
        "LipoaffinityIndex",
        "dipole_moment",
        "nRing",
        "VABC",
        "LUMO",
        "HOMO",
        "nAtom",
        "nHeavyAtom",
        "topoDiameter",
    ]
    for c in needed_cols:
        out[c] = df_full[c] if c in df_full.columns else np.nan

    F_CONST = 96485.33212
    out["capacity_proxy_mAh_g"] = 2 * F_CONST / (3.6 * out["MW"]) 

    # Thresholds from data distributions.
    mw_pass = out["MW"] <= 500

    pol_mask = pd.Series(True, index=out.index)
    if "TopoPSA" in out.columns:
        lo, hi = out["TopoPSA"].quantile([0.05, 0.95])
        pol_mask &= out["TopoPSA"].between(lo, hi)
    if "LipoaffinityIndex" in out.columns:
        lo, hi = out["LipoaffinityIndex"].quantile([0.05, 0.95])
        pol_mask &= out["LipoaffinityIndex"].between(lo, hi)
    if "dipole_moment" in out.columns and out["dipole_moment"].notna().any():
        lo, hi = out["dipole_moment"].quantile([0.05, 0.95])
        pol_mask &= out["dipole_moment"].between(lo, hi)

    comp_mask = pd.Series(True, index=out.index)
    for col in ["nAtom", "nHeavyAtom", "VABC", "nRing", "topoDiameter"]:
        if col in out.columns and out[col].notna().any():
            comp_mask &= out[col] <= out[col].quantile(0.95)

    elec_mask = pd.Series(True, index=out.index)
    for col in ["LUMO", "HOMO"]:
        if col in out.columns and out[col].notna().any():
            lo, hi = out[col].quantile([0.01, 0.99])
            elec_mask &= out[col].between(lo, hi)

    pred_lo, pred_hi = np.quantile(y_train, [0.01, 0.99])
    elec_mask &= out["predicted_redox_potential"].between(pred_lo, pred_hi)

    out["passes_mw"] = mw_pass
    out["passes_polarity"] = pol_mask
    out["passes_complexity"] = comp_mask
    out["passes_electronic"] = elec_mask
    out["passes_all"] = (
        out["passes_mw"]
        & out["passes_polarity"]
        & out["passes_complexity"]
        & out["passes_electronic"]
    )

    out.to_csv(results_dir / "screening_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = out["passes_all"].map({True: "tab:green", False: "tab:gray"})
    ax.scatter(out["capacity_proxy_mAh_g"], out["predicted_redox_potential"], c=colors, alpha=0.8)
    ax.set_xlabel("Capacity proxy (mAh/g)")
    ax.set_ylabel("Predicted redox potential")
    ax.set_title("Screening: Predicted Potential vs Capacity Proxy")
    fig.tight_layout()
    fig.savefig(plots_dir / "screening_scatter.png", dpi=300)
    plt.close(fig)

    print("\\n=== SCREENING COUNTS ===")
    for col in ["passes_mw", "passes_polarity", "passes_complexity", "passes_electronic", "passes_all"]:
        print(f"{col}: {int(out[col].sum())}")

    top20 = (
        out[out["passes_all"]]
        .sort_values("predicted_redox_potential", ascending=False)
        .head(20)
    )
    print("\\nTop 20 candidates (passes all):")
    print(
        top20[
            [
                NAME_COL,
                "predicted_redox_potential",
                "MW",
                "TopoPSA",
                "LipoaffinityIndex",
                "dipole_moment",
                "nRing",
                "VABC",
                "LUMO",
                "HOMO",
                "capacity_proxy_mAh_g",
            ]
        ]
    )

    return out


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).resolve().parents[1]
    source_data_dir = Path("/Users/harryvoorhis/Downloads/org-redox-dataset-main/datasets")

    dirs = ensure_dirs(base_dir)
    for png in dirs["plots"].glob("*.png"):
        png.unlink()
    data_paths = copy_datasets(source_data_dir, dirs["data"])

    required = "DatasetQuinonesFiltered.csv"
    if required not in data_paths:
        raise FileNotFoundError(f"Missing required dataset: {required}")

    df_full, df_tr, df_tst = get_train_test(data_paths)

    print_eda(df_full, dirs["plots"])

    full_features, dropped_corr = choose_numeric_features(df_tr, df_tst)
    feature_sets, impute_log = build_feature_sets(df_full, df_tr, df_tst, full_features)
    impute_log.to_csv(dirs["data"] / "preprocessing_log.csv", index=False)

    y_tr = df_tr[TARGET_COL]
    y_tst = df_tst[TARGET_COL]

    results_df, best_objects = evaluate_feature_sets(feature_sets, y_tr, y_tst, dirs["plots"])
    results_df.to_csv(dirs["results"] / "model_comparison.csv", index=False)

    # Best per model across feature sets for CV + scatter plots.
    for model_name, cv_plot in [
        ("Ridge", "cv_ridge.png"),
        ("RandomForest", "cv_random_forest.png"),
        ("GradientBoosting", "cv_gradient_boosting.png"),
    ]:
        subset = results_df[results_df["model"] == model_name].sort_values("test_r2", ascending=False)
        if subset.empty:
            continue
        best_row = subset.iloc[0]
        key = f"{model_name}::{best_row['feature_set']}"
        plot_cv_curve(best_objects[key]["cv_results"], model_name, dirs["plots"] / cv_plot)

    for model_name, scatter_plot in [
        ("Ridge", "ridge_test_scatter.png"),
        ("RandomForest", "random_forest_test_scatter.png"),
        ("GradientBoosting", "gradient_boosting_test_scatter.png"),
    ]:
        subset = results_df[results_df["model"] == model_name].sort_values("test_r2", ascending=False)
        if subset.empty:
            continue
        best_row = subset.iloc[0]
        key = f"{model_name}::{best_row['feature_set']}"
        plot_test_scatter(
            y_tst,
            best_objects[key]["y_pred_test"],
            f"{model_name} Predicted vs Actual",
            dirs["plots"] / scatter_plot,
        )

    plot_model_comparison(results_df, "test_r2", dirs["plots"] / "model_comparison_r2.png")
    plot_model_comparison(results_df, "test_rmse", dirs["plots"] / "model_comparison_rmse.png")

    best_overall = results_df.sort_values("test_r2", ascending=False).iloc[0]
    best_key = f"{best_overall['model']}::{best_overall['feature_set']}"
    best_obj = best_objects[best_key]
    best_fs = best_obj["feature_set"]
    best_est = best_obj["estimator"]

    print("\\n=== BEST OVERALL MODEL ===")
    print(best_overall)

    feature_ranking_df = pd.DataFrame()
    model = best_est.named_steps["model"]
    feature_names = best_fs.train_df.columns.tolist()

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        feature_ranking_df = (
            imp.abs()
            .sort_values(ascending=False)
            .head(50)
            .reset_index()
            .rename(columns={"index": "feature", 0: "importance"})
        )
        feature_ranking_df["impurity_rank"] = np.arange(1, len(feature_ranking_df) + 1)
    elif hasattr(model, "coef_"):
        imp = pd.Series(np.abs(model.coef_), index=feature_names)
        feature_ranking_df = (
            imp.sort_values(ascending=False)
            .head(50)
            .reset_index()
            .rename(columns={"index": "feature", 0: "coef_abs"})
        )
        feature_ranking_df["coef_rank"] = np.arange(1, len(feature_ranking_df) + 1)

    # Importance stability across feature sets (best model type retrained).
    stability_rows = []
    best_model_name = best_overall["model"]
    model_template, _, scale_flag = get_models_and_grids()[best_model_name]
    for fs in feature_sets:
        est = build_pipeline(clone(model_template), scale=scale_flag)
        est.fit(fs.train_df, y_tr)
        m = est.named_steps["model"]
        if hasattr(m, "feature_importances_"):
            imp = pd.Series(m.feature_importances_, index=fs.train_df.columns)
        elif hasattr(m, "coef_"):
            imp = pd.Series(np.abs(m.coef_), index=fs.train_df.columns)
        else:
            continue
        top_feats = imp.sort_values(ascending=False).head(25).index.tolist()
        for rank, feat in enumerate(top_feats, start=1):
            stability_rows.append({"feature_set": fs.name, "feature": feat, "rank": rank})

    stability_df = pd.DataFrame(stability_rows)
    if not stability_df.empty:
        stable_counts = stability_df.groupby("feature")["feature_set"].nunique().reset_index()
        stable_counts = stable_counts.rename(columns={"feature_set": "n_feature_sets_present"})
        feature_ranking_df = feature_ranking_df.merge(stable_counts, on="feature", how="outer")

    rank_cols = [c for c in ["coef_rank", "impurity_rank", "permutation_rank"] if c in feature_ranking_df.columns]
    if rank_cols:
        feature_ranking_df = feature_ranking_df.sort_values(
            by=rank_cols, na_position="last"
        ).reset_index(drop=True)

    feature_ranking_df.to_csv(dirs["results"] / "feature_ranking.csv", index=False)

    # Error analysis on best model.
    err_df, residual_stats, by_bin = residual_analysis(
        df_tst[NAME_COL], y_tst, best_obj["y_pred_test"], dirs["plots"]
    )
    print("\\n=== RESIDUAL STATS ===")
    print(residual_stats)
    print("\\n10 worst predictions:")
    print(err_df.head(10))

    class_df = descriptor_class_analysis(df_tst, y_tst, best_obj["y_pred_test"], dirs["plots"])
    if not class_df.empty:
        print("\\n=== ERROR BY DESCRIPTOR-DEFINED CLASS ===")
        print(class_df)

    screening_df = screening_pipeline(
        df_full,
        next(fs for fs in feature_sets if fs.name == best_fs.name),
        best_est,
        y_tr,
        dirs["results"],
        dirs["plots"],
    )

    # Final concise summary for paper workflow.
    print("\\n=== FINAL SUMMARY ===")
    print(
        f"Best model + feature set: {best_overall['model']} on {best_overall['feature_set']} | "
        f"Test RMSE={best_overall['test_rmse']:.4f}, MAE={best_overall['test_mae']:.4f}, R2={best_overall['test_r2']:.4f}"
    )

    if not feature_ranking_df.empty:
        top5 = feature_ranking_df.head(5)
        print("Top 5 important descriptors:")
        print(top5)

    if not class_df.empty:
        hardest = class_df.sort_values("MAE", ascending=False).iloc[0]
        easiest = class_df.sort_values("MAE", ascending=True).iloc[0]
        print(
            f"Hardest class: {hardest['class']} (MAE={hardest['MAE']:.4f}); "
            f"Easiest class: {easiest['class']} (MAE={easiest['MAE']:.4f})"
        )

    print(f"Molecules passing all filters: {int(screening_df['passes_all'].sum())}")
    top_candidate = screening_df[screening_df["passes_all"]].sort_values(
        "predicted_redox_potential", ascending=False
    )
    if not top_candidate.empty:
        t = top_candidate.iloc[0]
        print(
            "Best candidate: "
            f"{t[NAME_COL]} | predicted_redox_potential={t['predicted_redox_potential']:.4f}, "
            f"MW={t['MW']:.2f}, TopoPSA={t['TopoPSA']:.2f}, "
            f"LUMO={t['LUMO']:.4f}, HOMO={t['HOMO']:.4f}, "
            f"capacity_proxy_mAh_g={t['capacity_proxy_mAh_g']:.2f}"
        )


if __name__ == "__main__":
    main()
