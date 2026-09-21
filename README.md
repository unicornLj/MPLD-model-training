# MPLD model training and figures

Selected code for the Mongolian Plateau Livestock Density (MPLD) study.

## Contents

- **01_PVI_simulation/**: moving-window quantile random forest (QRF) fitting and annual PVI prediction.
- **02_GWRF_training/**: geographically weighted random forest (GWRF) training.
- **03_Figure/**: notebooks for PVI validation (Fig. 3), GWRF validation (Fig. 4), and bagh-scale validation (Fig. 5).

This release contains selected modelling and plotting code, not the complete processing workflow. Input preparation, validation runs, livestock raster allocation, input data, fitted models and generated figures are not included.

## Use

Use Python 3.10 with the packages in `requirements.txt`. Change the example paths under `E:/MPLD` to match your files. Use a separate output directory to avoid replacing existing results.

### 01 PVI simulation

`fit_predict_pvi.py` reads a prepared `pvi_training_samples.parquet` table with `year`, `AVI`, and the 13 columns listed in `FEATURES`. Prediction requires the reference grid, annual AVI masks, six annual predictors, three static predictors, and four climate-normal rasters defined at the top of the script. All rasters must be aligned and predictors valid within the prediction mask.

The script fits five-year moving-window QRF models and predicts the 0.90 quantile for 2000–2024. It writes model files and annual PVI rasters.

```bash
python 01_PVI_simulation/fit_predict_pvi.py
```

### 02 GWRF training

Edit `02_GWRF_training/config.yaml`. Supply:

- `county_year_features.parquet` in the configured work directory: `admin_id`, `log_SU_density`, and the 14 columns listed in `common.FEATURES`. Rows must already be restricted to the selected training counties.
- `county_model_domain_362.csv`: one row per target county (IDs 1–442), including `admin_id`, `centroid_x`, `centroid_y`, and `natural_grazing_training_county`. Coordinates must use the same projected, metre-based CRS.

The target is `log(1 + SU / effective grassland area in km²)`. The code retains the legacy feature name `VUR` for AD. Training uses 50 neighbouring counties, adaptive bisquare weights and 200-tree random forests. It saves 442 local models, neighbour weights and feature importances. It does not generate livestock rasters.

```bash
python 02_GWRF_training/train_gwrf.py --config 02_GWRF_training/config.yaml
```

### 03 Figure plotting

Open each notebook, edit the paths in its first code cell, and run cells in order. Fig. 5 also defines boundary and reference-grid paths in the mapping cell.

- Fig. 3 reads `annual_metrics.csv` and `overall_metrics.csv` for the `full_5fold_oof` protocol.
- Fig. 4 reads `random70_30_predictions.csv` and `county5fold_predictions.csv`, including observed/predicted log density and country.
- Fig. 5 reads `bagh_modeled_SU_2012_2023.csv` and `bagh_official_SU_2012_2023.csv`; the map also needs county boundaries and a reference raster.

These tables and geographic inputs are prepared separately. Notebook outputs have been cleared. Plotting code retains the source calculations and layouts; Fig. 3 annotations read the supplied summary values.

## Scope

This repository shares code for inspection during review. It does not claim to reproduce the entire dataset without separately prepared inputs. No software licence is assigned in this initial release.
