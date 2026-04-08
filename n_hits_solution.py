# ============================================================
# N-HiTS — 10 систематических экспериментов
# Phase 1 (1-4): n_freq_downsample при pool=[1,1,1], blocks=[4,4,4]
# Phase 2 (5-8): n_blocks при pool=[1,1,1], freq=[48,8,1]  (изоляция!)
# Phase 3 (9-10): лучший freq × лучший blocks
#
# Каждый эксперимент: CV_STEPS=400
# Победитель в конце: полная CV на FINAL_STEPS=800 → predict()
# ============================================================

import warnings, gc
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# ── Метрика ──
def wape_rbias(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.clip(np.asarray(y_pred, float), 0, None)
    s  = yt.sum() + 1e-9
    wape  = np.abs(yp - yt).sum() / s
    rbias = np.abs(yp.sum() / s - 1)
    return wape + rbias, wape, rbias

# ── Config ──
TRAIN_PATH     = "train_solo_track.parquet"
TEST_PATH      = "test_solo_track.parquet"
TARGET_COL     = "target_1h"
FORECAST_STEPS = 8
N_FOLDS        = 5
TOTAL_VAL_PTS  = N_FOLDS * FORECAST_STEPS
CONTEXT_LEN    = 2048
FREQ           = "30min"
INPUT_SIZE     = 336
BATCH_SIZE     = 64

CV_STEPS    = 400   # все 10 экспериментов — быстрый прогон
FINAL_STEPS = 800   # победитель переобучается здесь для честного CV + predict

# Предыдущий лучший результат из grid search — точка отсчёта
PREV_BEST_TOTAL = 0.3563  # ← замени на число из прошлого прогона, напр. 0.3142

if torch.backends.mps.is_available():
    ACCELERATOR = "mps"
elif torch.cuda.is_available():
    ACCELERATOR = "gpu"
else:
    ACCELERATOR = "cpu"
print(f"Device: {ACCELERATOR}")

# ── Data ──
train_df = pd.read_parquet(TRAIN_PATH)
test_df  = pd.read_parquet(TEST_PATH)
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"]  = pd.to_datetime(test_df["timestamp"])
train_df = train_df.sort_values(["route_id","timestamp"]).reset_index(drop=True)
test_df  = test_df.sort_values(["route_id","timestamp"]).reset_index(drop=True)
status_cols = sorted([c for c in train_df.columns if c in ["status_2","status_3","status_5"]])

FUTR_EXOG = ["hour_sin","hour_cos","dow_sin","dow_cos","is_weekend"]
HIST_EXOG = (status_cols + ["lag_48","lag_336"]) or None

def add_calendar_features(df, ts_col="ds"):
    ts = pd.to_datetime(df[ts_col])
    df = df.copy()
    df["hour_sin"]   = np.sin(2*np.pi*ts.dt.hour/24)
    df["hour_cos"]   = np.cos(2*np.pi*ts.dt.hour/24)
    df["dow_sin"]    = np.sin(2*np.pi*ts.dt.dayofweek/7)
    df["dow_cos"]    = np.cos(2*np.pi*ts.dt.dayofweek/7)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    return df

def add_lag_features(nf_df):
    nf_df = nf_df.sort_values(["unique_id","ds"]).copy()
    for lag in [48, 336]:
        nf_df[f"lag_{lag}"] = (
            nf_df.groupby("unique_id")["y"]
            .transform(lambda s, l=lag: s.shift(l).bfill().fillna(0))
        )
    return nf_df

def to_nf_df(df, min_len=32, context_len=CONTEXT_LEN):
    rows = []
    for route_id, grp in df.groupby("route_id"):
        grp = (
            grp.sort_values("timestamp")
            .set_index("timestamp")[[TARGET_COL]+status_cols]
            .asfreq(FREQ).interpolate(method="time")
            .bfill().ffill().tail(context_len).reset_index()
        )
        if len(grp) < min_len:
            continue
        grp.insert(0, "unique_id", route_id)
        grp = grp.rename(columns={"timestamp":"ds", TARGET_COL:"y"})
        rows.append(grp)
    nf_df = pd.concat(rows, ignore_index=True)
    nf_df = add_calendar_features(nf_df)
    nf_df = add_lag_features(nf_df)
    return nf_df

def make_futr_df(route_ids, train_df):
    rows = []
    for rid in route_ids:
        last_ts = train_df[train_df["route_id"]==rid]["timestamp"].max()
        for ts in pd.date_range(start=last_ts+pd.Timedelta("30min"),
                                periods=FORECAST_STEPS, freq=FREQ):
            rows.append({"unique_id": rid, "ds": ts})
    return add_calendar_features(pd.DataFrame(rows))

def get_pred_col(pred_df):
    for c in ["NHITS-median","NHITS-q-0.50","NHITS"]:
        if c in pred_df.columns:
            return c
    skip = {"unique_id","ds","y","cutoff"}
    num = [c for c in pred_df.columns if c not in skip
           and pd.api.types.is_numeric_dtype(pred_df[c])]
    med = [c for c in num if any(x in c for x in ["median","0.5","50"])]
    return (med or num)[0]

def build_preds_dict(pred_df, pred_col):
    return {rid: np.clip(g.sort_values("ds")[pred_col].values, 0, None)
            for rid, g in pred_df.groupby("unique_id")}

def make_nhits(cfg, max_steps):
    return NHITS(
        h                  = FORECAST_STEPS,
        input_size         = INPUT_SIZE,
        loss               = MQLoss(level=[80]),
        n_freq_downsample  = cfg["n_freq_downsample"],
        n_pool_kernel_size = cfg["n_pool_kernel_size"],
        n_blocks           = cfg["n_blocks"],
        mlp_units          = [[512, 512]] * len(cfg["n_blocks"]),
        max_steps          = max_steps,
        batch_size         = BATCH_SIZE,
        accelerator        = ACCELERATOR,
        hist_exog_list     = HIST_EXOG,
        futr_exog_list     = FUTR_EXOG,
        scaler_type        = "robust",
        enable_progress_bar= True,
    )

def run_cv(cfg, train_nf, max_steps):
    """Запускает CV и возвращает dict с метриками."""
    nf = NeuralForecast(models=[make_nhits(cfg, max_steps)], freq=FREQ)
    cv_df = nf.cross_validation(
        df=train_nf, n_windows=N_FOLDS, step_size=FORECAST_STEPS, refit=True
    )
    pred_col = get_pred_col(cv_df)

    # fold mapping
    cuts = sorted(cv_df["cutoff"].unique())
    cv_df["fold"] = cv_df["cutoff"].map({c: i+1 for i, c in enumerate(cuts)})
    cv_df = cv_df.sort_values(["fold","unique_id","ds"]).reset_index(drop=True)
    cv_df["h"] = cv_df.groupby(["fold","unique_id"]).cumcount() + 1

    yt = cv_df["y"].values
    yp = np.clip(cv_df[pred_col].values, 0, None)
    tot, wape, rb = wape_rbias(yt, yp)
    calib = float(yt.sum() / (yp.sum() + 1e-9))
    yp_c  = np.clip(yp * calib, 0, None)
    tot_c, wape_c, rb_c = wape_rbias(yt, yp_c)

    # per-fold
    fold_rows = []
    for fold in range(1, N_FOLDS+1):
        fdf = cv_df[cv_df["fold"]==fold]
        t, w, r = wape_rbias(fdf["y"].values,
                              np.clip(fdf[pred_col].values, 0, None))
        fold_rows.append({"fold": fold, "wape": round(w,4),
                          "rbias": round(r,4), "total": round(t,4)})

    del nf, cv_df
    gc.collect()

    return {
        "wape_raw":    round(wape,  4),
        "rbias_raw":   round(rb,    4),
        "total_raw":   round(tot,   4),
        "wape_cal":    round(wape_c,4),
        "rbias_cal":   round(rb_c,  4),
        "total_cal":   round(tot_c, 4),
        "calib_scale": round(calib, 4),
        "fold_stats":  fold_rows,
    }

# ──────────────────────────────────────────────────────────────
# 10 ЭКСПЕРИМЕНТОВ
# ──────────────────────────────────────────────────────────────
# Phase 3 пре-планирована как freq_4_2_1 × best_blocks.
# После прогона фаз 1-2 можно скорректировать вручную.

EXPERIMENTS = [
    # ── Phase 1: freq (pool=[1,1,1], blocks=[4,4,4]) ──────────
    # {"phase": 1, "name": "p1_freq_4_2_1",
    #  "n_freq_downsample": [4,2,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [4,4,4],
    #  "note": "Честная иерархия для h=8: 2/4/8 контрол. точек"},

    {"phase": 1, "name": "p1_freq_8_4_1",
     "n_freq_downsample": [8,4,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [4,4,4],
     "note": "4ч/2ч/30мин — совпадает с сильнейшей сезонностью (4h_8)"},

    {"phase": 1, "name": "p1_freq_2_1_1",
     "n_freq_downsample": [2,1,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [4,4,4],
     "note": "Два стека на полном разрешении, один — полугрубый"},

    {"phase": 1, "name": "p1_freq_1_1_1",
     "n_freq_downsample": [1,1,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [4,4,4],
     "note": "Нет иерархии вообще — 3 параллельных MLP на одном сигнале"},

    # ── Phase 2: n_blocks (pool=[1,1,1], freq=[48,8,1]) ───────
    # freq намеренно остаётся baseline для изоляции эффекта блоков
    {"phase": 2, "name": "p2_blk_2_4_8",
     "n_freq_downsample": [48,8,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [2,4,8],
     "note": "Больше параметров в short-term стеке (8 блоков)"},

    {"phase": 2, "name": "p2_blk_6_6_6",
     "n_freq_downsample": [48,8,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [6,6,6],
     "note": "Просто больше ёмкости везде (+50% блоков)"},

    {"phase": 2, "name": "p2_blk_2_2_8",
     "n_freq_downsample": [48,8,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [2,2,8],
     "note": "Экстремальный акцент на коротком стеке"},

    {"phase": 2, "name": "p2_blk_8_4_2",
     "n_freq_downsample": [48,8,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [8,4,2],
     "note": "Инверсия: акцент на длинном стеке — проверка обратной гипотезы"},

    # ── Phase 3: combo (best_freq × best_blocks) ──────────────
    # Пре-планировано как наиболее перспективные пары
    {"phase": 3, "name": "p3_4_2_1_blk248",
     "n_freq_downsample": [4,2,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [2,4,8],
     "note": "Честная иерархия + акцент блоков на коротком стеке"},

    {"phase": 3, "name": "p3_4_2_1_blk666",
     "n_freq_downsample": [4,2,1], "n_pool_kernel_size": [1,1,1], "n_blocks": [6,6,6],
     "note": "Честная иерархия + увеличенная ёмкость везде"},
]

# ── Данные ──
train_nf = to_nf_df(
    train_df,
    min_len     = TOTAL_VAL_PTS + INPUT_SIZE + 1,
    context_len = CONTEXT_LEN,
)
print(f"NF train: {train_nf.shape}  routes: {train_nf['unique_id'].nunique()}\n")

# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────
all_results   = []
best_so_far   = {"total_raw": PREV_BEST_TOTAL or float("inf"), "name": "prev_best"}
current_phase = None

for i, cfg in enumerate(EXPERIMENTS, 1):

    # Заголовок фазы
    if cfg["phase"] != current_phase:
        current_phase = cfg["phase"]
        phase_titles = {
            1: "PHASE 1 — n_freq_downsample  (pool=[1,1,1], blocks=[4,4,4] fixed)",
            2: "PHASE 2 — n_blocks  (pool=[1,1,1], freq=[48,8,1] fixed)",
            3: "PHASE 3 — Combo: best_freq × best_blocks",
        }
        print(f"\n{'='*65}")
        print(phase_titles[current_phase])
        print("="*65)

    print(f"\n[{i:02d}/10] {cfg['name']}")
    print(f"  freq={cfg['n_freq_downsample']}  "
          f"pool={cfg['n_pool_kernel_size']}  "
          f"blocks={cfg['n_blocks']}")
    print(f"  {cfg['note']}")

    metrics = run_cv(cfg, train_nf, CV_STEPS)

    # per-fold вывод
    for fr in metrics["fold_stats"]:
        print(f"  Fold {fr['fold']}  WAPE={fr['wape']:.4f}  "
              f"|RBias|={fr['rbias']:.4f}  Total={fr['total']:.4f}")

    print(f"\n  ── RAW    WAPE={metrics['wape_raw']:.4f}  "
          f"|RBias|={metrics['rbias_raw']:.4f}  "
          f"Total={metrics['total_raw']:.4f}")
    print(f"  ── CALIB  WAPE={metrics['wape_cal']:.4f}  "
          f"|RBias|={metrics['rbias_cal']:.4f}  "
          f"Total={metrics['total_cal']:.4f}  "
          f"scale={metrics['calib_scale']:.4f}")

    row = {"#": i, "name": cfg["name"], "phase": cfg["phase"],
           "freq": str(cfg["n_freq_downsample"]),
           "blocks": str(cfg["n_blocks"]), **metrics}
    del row["fold_stats"]
    all_results.append(row)

    # Обновляем победителя
    if metrics["total_raw"] < best_so_far["total_raw"]:
        best_so_far = {"name": cfg["name"], **metrics, "cfg": cfg}
        print(f"\n  🏆 NEW BEST!  Total={metrics['total_raw']:.4f}  "
              f"(prev: {all_results[-2]['total_raw'] if len(all_results)>1 else PREV_BEST_TOTAL:.4f})")
    else:
        delta = metrics["total_raw"] - best_so_far["total_raw"]
        print(f"\n  Current best: {best_so_far['name']}  "
              f"Total={best_so_far['total_raw']:.4f}  "
              f"(this: {metrics['total_raw']:.4f}, Δ={delta:+.4f})")

# ──────────────────────────────────────────────────────────────
# СВОДНАЯ ТАБЛИЦА
# ──────────────────────────────────────────────────────────────
results_df = (
    pd.DataFrame(all_results)
    .sort_values("total_raw")
    [["#","name","phase","freq","blocks","wape_raw","rbias_raw","total_raw","total_cal","calib_scale"]]
)
results_df.to_csv("nhits_exp10_results.csv", index=False)

print("\n\n" + "="*65)
print("LEADERBOARD — все 10 экспериментов (CV_STEPS=400)")
print("="*65)
print(results_df.to_string(index=False))

best_cfg = best_so_far["cfg"]
print(f"\n🏆 WINNER:  {best_so_far['name']}")
print(f"   freq={best_cfg['n_freq_downsample']}  "
      f"pool={best_cfg['n_pool_kernel_size']}  "
      f"blocks={best_cfg['n_blocks']}")
print(f"   total_raw={best_so_far['total_raw']:.4f}")

# ──────────────────────────────────────────────────────────────
# WINNER — ПОЛНАЯ CV (FINAL_STEPS=800)
# Эти метрики сравнимы с прошлым grid search (тоже FINAL_STEPS)
# ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"WINNER FULL CV  (FINAL_STEPS={FINAL_STEPS})")
print(f"Конфиг: {best_so_far['name']}")
print("="*65)

winner_metrics = run_cv(best_cfg, train_nf, FINAL_STEPS)

print(f"\nPer-fold:")
for fr in winner_metrics["fold_stats"]:
    print(f"  Fold {fr['fold']}  WAPE={fr['wape']:.4f}  "
          f"|RBias|={fr['rbias']:.4f}  Total={fr['total']:.4f}")

print(f"\n── RAW    WAPE={winner_metrics['wape_raw']:.4f}  "
      f"|RBias|={winner_metrics['rbias_raw']:.4f}  "
      f"Total={winner_metrics['total_raw']:.4f}")
print(f"── CALIB  WAPE={winner_metrics['wape_cal']:.4f}  "
      f"|RBias|={winner_metrics['rbias_cal']:.4f}  "
      f"Total={winner_metrics['total_cal']:.4f}  "
      f"scale={winner_metrics['calib_scale']:.4f}")

calib_scale = winner_metrics["calib_scale"]

# Сохраняем финальные метрики победителя
pd.DataFrame([{
    "name": best_so_far["name"],
    "steps": FINAL_STEPS,
    **{k: v for k, v in winner_metrics.items() if k != "fold_stats"},
}]).to_csv("nhits_winner_full_cv.csv", index=False)

# ──────────────────────────────────────────────────────────────
# ФИНАЛЬНЫЙ PREDICT
# ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"FINAL PREDICT  (FINAL_STEPS={FINAL_STEPS})")
print("="*65)

train_nf_full  = to_nf_df(train_df, min_len=INPUT_SIZE+1, context_len=CONTEXT_LEN)
test_route_ids = test_df["route_id"].unique().tolist()
print(f"Routes: {len(test_route_ids)}  Fitting on full train data...")

nf_full = NeuralForecast(models=[make_nhits(best_cfg, FINAL_STEPS)], freq=FREQ)
nf_full.fit(train_nf_full)

futr_df      = make_futr_df(test_route_ids, train_df)
test_pred_df = nf_full.predict(futr_df=futr_df)
test_pred_col = get_pred_col(test_pred_df)
test_preds    = build_preds_dict(test_pred_df, test_pred_col)

predictions_raw = {}
for route_id in tqdm(test_route_ids, desc="Build submission"):
    route_test = test_df[test_df["route_id"]==route_id].sort_values("timestamp")
    preds = test_preds.get(route_id, np.zeros(FORECAST_STEPS))
    for j, (_, row) in enumerate(route_test.iterrows()):
        pred = float(preds[j]) if j < len(preds) else float(preds[-1])
        predictions_raw[row["id"]] = max(0.0, pred)

name = best_so_far["name"]
submission_raw = (
    pd.DataFrame(list(predictions_raw.items()), columns=["id","y_pred"])
    .sort_values("id").reset_index(drop=True)
)
submission_cal = submission_raw.copy()
submission_cal["y_pred"] = np.clip(submission_cal["y_pred"] * calib_scale, 0, None)

submission_raw.to_csv(f"submission_{name}_raw.csv",        index=False)
submission_cal.to_csv(f"submission_{name}_calibrated.csv", index=False)

print(f"\n✅ nhits_exp10_results.csv")
print(f"✅ nhits_winner_full_cv.csv")
print(f"✅ submission_{name}_raw.csv")
print(f"✅ submission_{name}_calibrated.csv")
print(f"\n── Raw ──\n{submission_raw['y_pred'].describe().round(2)}")
print(f"\n── Calibrated (scale={calib_scale:.4f}) ──\n{submission_cal['y_pred'].describe().round(2)}")
