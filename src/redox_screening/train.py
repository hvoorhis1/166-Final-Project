from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import DatasetSpec, build_precomputed_feature_matrix, load_dataset
from .models import build_models, crossval_summary, regression_metrics
from .screening import apply_candidate_filter, compute_screening_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train redox potential ML models and generate screening outputs"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to CSV dataset or org-redox-dataset-main directory",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for run outputs")
    parser.add_argument("--smiles-col", default="smiles", help="Preferred SMILES column")
    parser.add_argument("--target-col", default="deltaE_V", help="Preferred target column")
    parser.add_argument("--id-col", default="Name", help="Preferred ID/name column")
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of highest-predicted candidates to export per model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = DatasetSpec(
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        id_col=args.id_col,
    )
    loaded = load_dataset(args.data_path, spec)
    df = loaded.df
    target_col = loaded.target_col

    y = df[target_col].to_numpy(dtype=float)
    if loaded.mode == "smiles":
        from .features import FingerprintConfig, build_feature_sets

        feature_sets = build_feature_sets(
            df[loaded.smiles_col],
            FingerprintConfig(
                radius=args.fingerprint_radius,
                n_bits=args.fingerprint_bits,
            ),
        )
    else:
        x, names = build_precomputed_feature_matrix(
            df,
            target_col=target_col,
            id_col=loaded.id_col,
        )
        feature_sets = {"precomputed_descriptor": (x, names)}
    models = build_models(random_state=args.random_state, n_jobs=args.n_jobs)

    metric_rows: list[dict[str, float | str]] = []
    cv_rows: list[dict[str, float | str]] = []

    for representation, (x, feature_names) in feature_sets.items():
        idx_train, idx_test = train_test_split(
            np.arange(len(df)),
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=True,
        )
        x_train, x_test = x[idx_train], x[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        for model_spec in models:
            model = model_spec.estimator
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            metrics = regression_metrics(y_test, y_pred)
            metric_rows.append(
                {
                    "representation": representation,
                    "model": model_spec.name,
                    **metrics,
                }
            )

            cv = crossval_summary(model, x_train, y_train, random_state=args.random_state)
            cv_rows.append(
                {
                    "representation": representation,
                    "model": model_spec.name,
                    **cv,
                }
            )

            test_df = df.iloc[idx_test].copy()
            test_df["y_true"] = y_test
            test_df["y_pred"] = y_pred
            screen = compute_screening_flags(test_df, smiles_col=loaded.smiles_col)
            pred_out = pd.concat([test_df.reset_index(drop=True), screen], axis=1)
            pred_out.to_csv(
                out_dir / f"predictions_{representation}_{model_spec.name}.csv",
                index=False,
            )

            filtered = apply_candidate_filter(pred_out)
            ranked = filtered.sort_values("y_pred", ascending=False).head(args.top_k)
            ranked.to_csv(
                out_dir / f"top_candidates_{representation}_{model_spec.name}.csv",
                index=False,
            )

            if (
                representation in {"descriptor", "precomputed_descriptor"}
                and model_spec.name == "random_forest"
            ):
                rf_model = model.named_steps["model"]
                importances = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": rf_model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                importances.to_csv(
                    out_dir / "feature_importance_descriptor_random_forest.csv",
                    index=False,
                )

    pd.DataFrame(metric_rows).sort_values(["representation", "rmse"]).to_csv(
        out_dir / "metrics.csv", index=False
    )
    pd.DataFrame(cv_rows).sort_values(["representation", "cv_rmse_mean"]).to_csv(
        out_dir / "cv_metrics.csv", index=False
    )

    print(f"Completed training and screening. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
