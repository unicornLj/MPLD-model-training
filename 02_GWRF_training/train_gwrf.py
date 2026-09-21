from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import (
    FEATURES,
    N_COUNTIES,
    feature_matrix,
    load_domain,
    local_neighbors,
    make_output_dirs,
    new_rf,
    read_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3: train 442 final local GWRF models.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    cfg = read_config(args.config)
    make_output_dirs(cfg)
    features = pd.read_parquet(
        Path(cfg["outputs"]["work"]) / "county_year_features.parquet"
    )
    domain = load_domain(cfg)
    train_ids = np.sort(features["admin_id"].unique())
    models = {}
    neighbor_rows = []
    importance_rows = []

    for target_id in range(1, N_COUNTIES + 1):
        neighbor_ids, county_weights = local_neighbors(
            target_id, train_ids, domain, cfg["model"]["neighbors"]
        )
        weights = pd.Series(county_weights, index=neighbor_ids)
        local_train = features[features["admin_id"].isin(neighbor_ids)]
        model = new_rf(cfg)
        model.fit(
            feature_matrix(local_train),
            local_train["log_SU_density"],
            sample_weight=local_train["admin_id"].map(weights),
        )
        models[target_id] = model
        for rank, (neighbor_id, weight) in enumerate(
            zip(neighbor_ids, county_weights), start=1
        ):
            neighbor_rows.append(
                {
                    "target_admin_id": target_id,
                    "neighbor_rank": rank,
                    "training_admin_id": int(neighbor_id),
                    "bisquare_weight": float(weight),
                }
            )
        for feature, importance in zip(FEATURES, model.feature_importances_):
            importance_rows.append(
                {
                    "target_admin_id": target_id,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
        if target_id % 25 == 0 or target_id == N_COUNTIES:
            print(f"trained {target_id}/{N_COUNTIES} local models")

    output = Path(cfg["outputs"]["models"])
    joblib.dump(
        {
            "models": models,
            "feature_names": FEATURES,
            "training_admin_ids": train_ids,
            "neighbors": cfg["model"]["neighbors"],
            "kernel": "adaptive_bisquare",
        },
        output / "GWRF_SU_model_bundle.joblib",
        compress=3,
    )
    pd.DataFrame(neighbor_rows).to_csv(output / "local_model_neighbors.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(output / "feature_importance_by_county.csv", index=False)
    (
        importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .to_csv(output / "feature_importance.csv", index=False)
    )
    print(f"Saved final model bundle to {output}")


if __name__ == "__main__":
    args = parse_args()
    main()
