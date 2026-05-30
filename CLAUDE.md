# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Wildberries hackathon forecasting project. The task is time series forecasting of warehouse route throughput (`target_1h`) for 1000 routes at 30-minute granularity, forecasting 8 steps ahead (4 hours total). The evaluation metric is **WAPE + |Relative Bias|** (lower is better; ~0.33 is competitive).

## Data

Two tracks (this repo uses **solo track**):
- `train_solo_track.parquet` — training data (~4.6M rows, routes 0–999, July–November 2025)
- `test_solo_track.parquet` — test data (8000 rows: 1000 routes × 8 forecast steps, starting 2025-11-01 11:00)

Schema:
- `route_id` (int), `timestamp` (30min intervals), `status_1..6` (current/previous warehouse statuses), `target_1h` (target)
- `status_1/2/3` = current-period statuses; `status_4/5/6` = prior-period statuses
- Test set contains `id`, `route_id`, `timestamp` only; submissions use `id` + `y_pred`

Note: `status_2`, `status_3`, `status_5` are the most predictive status columns (others are less useful).

## Running Experiments

All work is done in Jupyter notebooks or Python scripts:

```bash
# Run the N-HiTS systematic experiment script
python n_hits_solution.py

# Launch Jupyter for interactive work
jupyter notebook wb_hack.ipynb
jupyter notebook ensemble_catboost_added_fixed_ridge_nan.ipynb
jupyter notebook baseline_template.ipynb
```

Hardware auto-detection in scripts: uses MPS (Apple Silicon) → GPU (CUDA) → CPU.

## Metric

```python
def wape_rbias(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.clip(np.asarray(y_pred, float), 0, None)
    s  = yt.sum() + 1e-9
    wape  = np.abs(yp - yt).sum() / s
    rbias = np.abs(yp.sum() / s - 1)
    return wape + rbias, wape, rbias
```

Predictions must be clipped to ≥ 0. Always apply a global calibration scale after CV to correct systematic over/under-prediction.

## Architecture Overview

### Models Explored

1. **N-HiTS** (`n_hits_solution.py`, `wb_hack.ipynb`) — neural hierarchical interpolation for time series. Uses `neuralforecast`. Key hyperparameters: `n_freq_downsample=[48,8,1]`, `n_pool_kernel_size=[1,1,1]`, `n_blocks=[8,4,2]` (best config found: `p2_blk_8_4_2`). Uses `MQLoss(level=[80])`, `INPUT_SIZE=336`, `CONTEXT_LEN=2048`, `FORECAST_STEPS=8`.

2. **LightGBM ensemble** (`ensemble_catboost_added_fixed_ridge_nan.ipynb`) — direct multi-step prediction, one model per forecast step. Multiple specs with varying `train_days` (5–14), objectives (Poisson, MAE), and decay weights. Blended per-horizon with calibration.

3. **CatBoost** — added to LightGBM ensemble as `cat_poisson_14d` spec.

4. **Ridge** — linear baseline in the ensemble.

5. **SARIMA/ARIMAX/SARIMAX** — classical per-route models (`wb_hack.ipynb`).

6. **Foundation models** — Chronos-2 (finetuned), Lag-Llama, TimesFM, TiRex, Sundial — inference via HuggingFace.

### Feature Engineering (LightGBM path)

- **Time features**: hour_sin/cos, dow_sin/cos, half_hour_idx, is_weekend
- **Lag features**: lags at [1,2,3,6,12,24,48] × 30min steps for all status and target cols
- **Rolling means**: windows [2,4,8,16,48] on all signal columns
- **Status aggregates**: status_current_sum (1+2+3), status_prev_sum (4+5+6), ratio
- **Route seasonal features**: route × half_hour_idx mean, route × dayofweek × half_hour_idx mean
- **Global timestamp features**: global sums and per-route share ratios
- **WB promo features**: 11.11 sale window flags, days_to_1111, WB birthday (Oct 7–20), back-to-school (Aug 15–Sep 1)

### N-HiTS Feature Engineering

- `futr_exog`: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`
- `hist_exog`: `status_2`, `status_3`, `status_5`, `lag_48`, `lag_336`
- Data formatted via `to_nf_df()` → NeuralForecast's `unique_id/ds/y` schema

### CV Strategy

- N-HiTS: rolling-window CV with `N_FOLDS=5`, `FORECAST_STEPS=8`, `refit=True`
- LightGBM: time-based split — fit window → `valid_early` (1 day before last) → `valid_calib` (last 12h) → test
- Calibration: compute global scale `y_true.sum() / y_pred.sum()` on calibration slice, apply to test predictions

### Submission Format

```python
submission = pd.DataFrame({"id": [...], "y_pred": [...]})
submission.to_csv("submission_<name>.csv", index=False)
```

Files named `submission_<model>_raw.csv` (uncalibrated) and `submission_<name>_calibrated.csv` (after scale).

## Key Files

| File | Purpose |
|------|---------|
| `wb_hack.ipynb` | **Main notebook** — **Cell 102** is the winning 10-seed N-HiTS ensemble (final submission) |
| `n_hits_solution.py` | Systematic N-HiTS architecture search (10 experiments: freq hierarchy + n_blocks phases) |
| `baseline_template.ipynb` | Clean template with data loading, EDA, and metric |
| `ensemble_catboost_added_fixed_ridge_nan.ipynb` | LightGBM + CatBoost + Ridge ensemble exploration (not the winner) |

### Final Solution (Cell 102 in wb_hack.ipynb)

Three-stage pipeline:
1. **Per-seed calibration CV** (400 steps, 1 window) → compute `calib_scale = sum(y_true)/sum(y_pred)` per seed
2. **Full train + predict** (800 steps) per seed → apply `calib_scale`
3. **Mean** across 10 calibrated seeds → `submission_p2_blk_8_4_2_ensemble10_calibrated_mean.csv`

Best architecture: `p2_blk_8_4_2` — `n_freq_downsample=[48,8,1]`, `n_blocks=[8,4,2]`, `INPUT_SIZE=336`, `CONTEXT_LEN=2048`, `MQLoss(level=[80])`, `scaler_type="robust"`.

## Dependencies

```python
# Core
pandas, numpy, pyarrow, matplotlib, seaborn, scikit-learn

# Gradient boosting
lightgbm, catboost

# Neural forecasting
neuralforecast, torch, pytorch-lightning

# Foundation models
transformers, huggingface_hub, chronos-forecasting, lag-llama

# Hyperparameter tuning
optuna
```
