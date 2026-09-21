"""Train moving-window QRF models from pixel-year samples and predict annual PVI.

Fit one model per five-year window. Change QUANTILE to generate q90, q95, or other
quantile products; adjust paths and years for other regions with aligned input rasters.

This script fits and saves models and predicts PVI across the region; spatial cross-validation is separate.
"""

from contextlib import ExitStack
from pathlib import Path
import logging
import math
import os
import sys

gdal_data = Path(sys.prefix) / "Library" / "share" / "gdal"
if gdal_data.exists():
    os.environ.setdefault("GDAL_DATA", str(gdal_data))

import joblib
import numpy as np
import pandas as pd
import rasterio
from quantile_forest import RandomForestQuantileRegressor
from rasterio.windows import Window


# =============================================================================
# 1. Input and output paths
# =============================================================================

REFERENCE = Path(
    r"E:\MPLD\geometry\M1_IM2_1km.tif"
)
AVI_ROOT = Path(r"E:\MPLD\AVI_2000_2024")
ERA_ROOT = Path(r"E:\MPLD\ERA5_PVI_annual\1km_clipped_aligned")
SIMPLE_ROOT = Path(r"E:\MPLD\PVI\PVI_QRF_V3_q90")
TRAINING_SAMPLES = SIMPLE_ROOT / "samples" / "pvi_training_samples.parquet"
NORMAL_ROOT = SIMPLE_ROOT / "climate_normals"

DYNAMIC_INPUTS = {
    "P_cold": (ERA_ROOT / "P_cold", "P_cold_10to4_{year}.tif"),
    "P_warm": (ERA_ROOT / "P_warm", "P_warm_5to9_{year}.tif"),
    "GDD5": (ERA_ROOT / "GDD5", "GDD5_5to9_{year}.tif"),
    "VPD": (ERA_ROOT / "VPD", "VPD_5to9_{year}.tif"),
    "Rs": (ERA_ROOT / "Rs", "Rs_5to9_{year}.tif"),
    "SMroot": (ERA_ROOT / "SMroot", "SMroot_5to9_{year}.tif"),
}

# Use gap-filled copies of the static predictors for full-region prediction.
STATIC_INPUTS = {
    "DEM": Path(r"E:\MPLD\PVI\PVI_QRF_V3_q90\01_preprocessed\production_static_filled\DEM_nearest_filled_M1_IM2_1km.tif"),
    "Slope": Path(r"E:\MPLD\PVI\PVI_QRF_V3_q90\01_preprocessed\production_static_filled\Slope_nearest_filled_M1_IM2_1km.tif"),
    "AWC": Path(r"E:\MPLD\SU\data\MP_AWC_1km.tif"),
}


# =============================================================================
# 2. Method parameters
# =============================================================================

REGION_NAME = "Mongolian_Plateau"
START_YEAR = 2000
END_YEAR = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))
WINDOW_YEARS = 5
QUANTILE = 0.90
RANDOM_SEED = 20260815
N_JOBS = 8

FEATURES = [
    "P_cold", "P_warm", "GDD5", "VPD", "Rs", "SMroot",
    "DEM", "Slope", "AWC",
    "P_warm_clim", "GDD5_clim", "VPD_clim", "SMroot_clim",
]

MODEL_PARAMETERS = {
    "n_estimators": 200,
    "min_samples_leaf": 10,
    "max_samples_leaf": 50,
    "max_features": 0.7,
    "max_depth": 24,
    "bootstrap": True,
    "max_samples": 0.75,
    "random_state": RANDOM_SEED,
    "n_jobs": N_JOBS,
}

NODATA = -9999.0
STRIP_HEIGHT = 128
PREDICTION_CHUNK = 100_000


# =============================================================================
# 3. Functions
# =============================================================================

def target_years_by_window():
    """Map each window start year to the target years that use its model."""
    half_window = WINDOW_YEARS // 2
    last_window_start = END_YEAR - WINDOW_YEARS + 1
    mapping = {start: [] for start in range(START_YEAR, last_window_start + 1)}

    for year in YEARS:
        start = min(max(year - half_window, START_YEAR), last_window_start)
        mapping[start].append(year)

    return mapping


def read_window(source, window):
    band = source.read(1, window=window, masked=True)
    values = np.asarray(band.data, dtype=np.float32)
    valid = (~np.ma.getmaskarray(band)) & np.isfinite(values)
    return values, valid


def predict_one_year(model, year, window_start, pvi_dir):
    """Read the 13 predictors for a target year and predict PVI in chunks within the AVI mask."""
    q_label = f"q{int(round(QUANTILE * 100)):02d}"
    avi_path = AVI_ROOT / f"AVI_{year}.tif"
    output_path = pvi_dir / f"PVI_{q_label}_{year}.tif"

    predictor_paths = {
        **{
            name: root / template.format(year=year)
            for name, (root, template) in DYNAMIC_INPUTS.items()
        },
        **STATIC_INPUTS,
        **{
            name: NORMAL_ROOT / f"{name}_{START_YEAR}_{END_YEAR}.tif"
            for name in ("P_warm_clim", "GDD5_clim", "VPD_clim", "SMroot_clim")
        },
    }

    with rasterio.open(REFERENCE) as reference:
        profile = reference.profile.copy()
    profile.update(
        driver="GTiff", dtype="float32", count=1, nodata=NODATA,
        compress="deflate", predictor=3, tiled=True,
        blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER",
    )

    with ExitStack() as stack:
        avi = stack.enter_context(rasterio.open(avi_path))
        predictors = {
            name: stack.enter_context(rasterio.open(path))
            for name, path in predictor_paths.items()
        }
        target = stack.enter_context(rasterio.open(output_path, "w", **profile))
        target.update_tags(
            REGION=REGION_NAME,
            QUANTILE=str(QUANTILE),
            TARGET_YEAR=str(year),
            MODEL_WINDOW=f"{window_start}-{window_start + WINDOW_YEARS - 1}",
        )

        total_strips = math.ceil(avi.height / STRIP_HEIGHT)
        for strip_number, row_start in enumerate(
            range(0, avi.height, STRIP_HEIGHT), start=1
        ):
            height = min(STRIP_HEIGHT, avi.height - row_start)
            window = Window(0, row_start, avi.width, height)
            _, avi_valid = read_window(avi, window)
            output = np.full(avi_valid.shape, NODATA, dtype=np.float32)

            if avi_valid.any():
                pixel_index = np.flatnonzero(avi_valid.ravel())
                feature_columns = []
                for feature in FEATURES:
                    values, _ = read_window(predictors[feature], window)
                    feature_columns.append(values.ravel()[pixel_index])

                x_predict = np.column_stack(feature_columns).astype(np.float32)
                prediction = np.empty(len(x_predict), dtype=np.float32)

                for start in range(0, len(x_predict), PREDICTION_CHUNK):
                    end = min(start + PREDICTION_CHUNK, len(x_predict))
                    prediction[start:end] = model.predict(
                        x_predict[start:end], quantiles=QUANTILE
                    ).astype(np.float32)

                output.ravel()[pixel_index] = prediction

            target.write(output, 1, window=window)
            logging.info(
                "Prediction progress for %s: %s/%s (%.1f%%)",
                year,
                strip_number,
                total_strips,
                strip_number / total_strips * 100,
            )

    logging.info(
        "Completed PVI for %s using the %s-%s model window",
        year, window_start, window_start + WINDOW_YEARS - 1,
    )


# =============================================================================
# 4. Main program
# =============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    q_label = f"q{int(round(QUANTILE * 100)):02d}"
    model_dir = SIMPLE_ROOT / "06_models" / q_label
    pvi_dir = SIMPLE_ROOT / "07_PVI"
    model_dir.mkdir(parents=True, exist_ok=True)
    pvi_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(TRAINING_SAMPLES)
    x_all = samples[FEATURES].to_numpy(dtype=np.float32)
    y_all = samples["AVI"].to_numpy(dtype=np.float32)
    sample_years = samples["year"].to_numpy(dtype=np.int16)

    for window_start, target_years in target_years_by_window().items():
        window_end = window_start + WINDOW_YEARS - 1
        use = (sample_years >= window_start) & (sample_years <= window_end)
        model = RandomForestQuantileRegressor(**MODEL_PARAMETERS)

        logging.info(
            "Training the %s-%s QRF with %s samples for target years %s",
            window_start, window_end, int(use.sum()), target_years,
        )
        model.fit(x_all[use], y_all[use])
        model_path = model_dir / f"QRF_{q_label}_{window_start}_{window_end}.joblib"
        joblib.dump(model, model_path, compress=3)

        for year in target_years:
            predict_one_year(model, year, window_start, pvi_dir)

    logging.info("Model directory: %s", model_dir)
    logging.info("PVI directory: %s", pvi_dir)


if __name__ == "__main__":
    main()
