from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor


N_COUNTIES = 442
FEATURES = [
    "PVI",
    "VUR",
    "AI",
    "GrassFraction",
    "Slope",
    "log_DWater",
    "log_DSettlement",
    "YearIndex",
    "GrassType_1",
    "GrassType_2",
    "GrassType_3",
    "GrassType_4",
    "Country_Mongolia",
    "Country_Inner_Mongolia",
]


def read_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    return config


def make_output_dirs(cfg: dict) -> None:
    for path in cfg["outputs"].values():
        Path(path).mkdir(parents=True, exist_ok=True)


def load_domain(cfg: dict) -> pd.DataFrame:
    frame = pd.read_csv(cfg["inputs"]["county_domain"])
    column = frame["natural_grazing_training_county"]
    if column.dtype != bool:
        frame["natural_grazing_training_county"] = (
            column.astype(str).str.lower().eq("true")
        )
    return frame.sort_values("admin_id").reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURES].to_numpy(np.float32)


def new_rf(cfg: dict) -> RandomForestRegressor:
    model = cfg["model"]
    return RandomForestRegressor(
        n_estimators=model["n_estimators"],
        max_features=model["max_features"],
        min_samples_leaf=model["min_samples_leaf"],
        random_state=model["random_seed"],
        n_jobs=-1,
    )


def local_neighbors(
    target_id: int, train_ids: np.ndarray, domain: pd.DataFrame, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    xy = domain.set_index("admin_id")[["centroid_x", "centroid_y"]]
    target = xy.loc[target_id].to_numpy(np.float64)
    candidate = xy.loc[train_ids].to_numpy(np.float64)
    distance = np.sqrt(np.square(candidate - target).sum(axis=1))
    order = np.argsort(distance, kind="stable")
    take = min(n_neighbors, len(train_ids))
    selected = order[:take]
    bandwidth = (
        distance[order[take]]
        if len(train_ids) > take
        else distance[selected].max() * 1.01
    )
    if bandwidth == 0:
        return train_ids[selected], np.ones(take, dtype=np.float64)
    ratio = distance[selected] / bandwidth
    weight = np.square(1.0 - np.square(ratio))
    return train_ids[selected], weight
