"""
Brisbane Water Quality — Time Series Forecasting Pipeline
Задача: Предсказание температуры воды (30-мин интервалы, горизонт 48 шагов = 24 часа)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_RAW   = Path("data/brisbane_water_quality.csv")
DATA_CLEAN = Path("data/prepared_temperature_ts.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Config ──────────────────────────────────────────────────────────────────
TARGET      = "Temperature"
FREQ        = "30min"
HORIZON     = 48          # 24 hours ahead
SEASON_LEN  = 48          # daily seasonality (48 half-hours)
N_WINDOWS   = 3
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# STEP 1 — Data Loading & Cleaning

def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    quality_cols = [c for c in df.columns if "[quality]" in c]
    df = df.drop(columns=quality_cols + ["Record number"])
    df = df.rename(columns={"Timestamp": "ds"})

    df = df.groupby("ds").mean().reset_index()

    full_idx = pd.date_range(df["ds"].min(), df["ds"].max(), freq=FREQ)
    df = df.set_index("ds").reindex(full_idx).rename_axis("ds").reset_index()

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].interpolate(method="linear", limit=4)
    df[num_cols] = df[num_cols].ffill().bfill()

    print(f"[load] Shape: {df.shape}  |  Missing after imputation: {df[TARGET].isna().sum()}")
    return df


# STEP 2 — Feature Engineering

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]        = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["day_of_year"] = df["ds"].dt.dayofyear
    df["month"]       = df["ds"].dt.month
    df["sin_hour"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_doy"]     = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"]     = np.cos(2 * np.pi * df["day_of_year"] / 365)
    for lag in [1, 2, 48, 96]:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    for w in [4, 48]:
        df[f"roll_mean_{w}"] = df[TARGET].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df[TARGET].shift(1).rolling(w).std()
    return df


# STEP 3 — Train/Test Split

def train_test_split_ts(df: pd.DataFrame, horizon: int):
    train = df.iloc[:-horizon].copy()
    test  = df.iloc[-horizon:].copy()
    print(f"[split] Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


# STEP 4 — Statistical Models

def run_statistical_models(train: pd.DataFrame, test: pd.DataFrame):
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA, AutoETS, AutoTheta, SeasonalNaive, Naive, MSTL
    )

    sf_train = train[["ds", TARGET]].copy()
    sf_train.insert(0, "unique_id", "brisbane_temp")
    sf_train = sf_train.rename(columns={TARGET: "y"})

    models = [
        SeasonalNaive(season_length=SEASON_LEN),
        Naive(),
        AutoARIMA(season_length=SEASON_LEN),
        AutoETS(season_length=SEASON_LEN),
        AutoTheta(season_length=SEASON_LEN),
        MSTL(season_length=[SEASON_LEN, SEASON_LEN*7]),
    ]

    sf = StatsForecast(models=models, freq=FREQ, n_jobs=-1)
    forecast = sf.forecast(h=HORIZON, df=sf_train).reset_index()

    cv = sf.cross_validation(
        df=sf_train, h=HORIZON, n_windows=N_WINDOWS, step_size=HORIZON
    )
    return forecast, cv, sf


# STEP 5 — ML Models

def run_ml_models(df_feat: pd.DataFrame):
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

    ml_df = df_feat[["ds", TARGET]].dropna().copy()
    ml_df.insert(0, "unique_id", "brisbane_temp")
    ml_df = ml_df.rename(columns={TARGET: "y"})

    models = [
        LGBMRegressor(n_estimators=200, learning_rate=0.05,
                      random_state=RANDOM_SEED, verbose=-1),
        RandomForestRegressor(n_estimators=100,
                              random_state=RANDOM_SEED, n_jobs=-1),
        xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                         random_state=RANDOM_SEED, verbosity=0),
    ]

    fcst = MLForecast(
        models=models,
        freq=FREQ,
        lags=[1, 2, 48, 96],
        lag_transforms={
            1:  [("rolling_mean", 4), ("rolling_std", 4)],
            48: [("rolling_mean", 48)],
        },
        date_features=["hour", "dayofweek", "month"],
        target_transforms=[Differences([1])],
    )
    fcst.fit(ml_df)
    forecast = fcst.predict(HORIZON)

    cv = fcst.cross_validation(
        df=ml_df, h=HORIZON, n_windows=N_WINDOWS, step_size=HORIZON
    )
    return forecast, cv, fcst


# STEP 6 — DL Models

def run_dl_models(train: pd.DataFrame):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, NBEATS, PatchTST

    nf_train = train[["ds", TARGET]].copy()
    nf_train.insert(0, "unique_id", "brisbane_temp")
    nf_train = nf_train.rename(columns={TARGET: "y"})

    INPUT_SIZE = HORIZON * 3

    models = [
        NHITS(h=HORIZON, input_size=INPUT_SIZE, max_steps=300,
              accelerator="cpu", random_seed=RANDOM_SEED),
        NBEATS(h=HORIZON, input_size=INPUT_SIZE, max_steps=300,
               accelerator="cpu", random_seed=RANDOM_SEED),
        PatchTST(h=HORIZON, input_size=INPUT_SIZE, max_steps=300,
                 accelerator="cpu", random_seed=RANDOM_SEED),
    ]

    nf = NeuralForecast(models=models, freq=FREQ)
    nf.fit(nf_train)
    forecast = nf.predict()
    return forecast, nf


# STEP 7 — Anomaly Detection

def detect_anomalies(df: pd.DataFrame) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    y   = df[TARGET].dropna().values
    idx = df[TARGET].dropna().index
    results = {}

    # 1. IQR
    Q1, Q3 = np.percentile(y, 25), np.percentile(y, 75)
    IQR     = Q3 - Q1
    iqr_mask = (y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)
    results["IQR"] = int(iqr_mask.sum())

    # 2. Z-score
    z_scores = np.abs(stats.zscore(y))
    results["Z-score (>3sigma)"] = int((z_scores > 3).sum())

    # 3. Isolation Forest
    scaler = StandardScaler()
    y_s    = scaler.fit_transform(y.reshape(-1, 1))
    iso    = IsolationForest(contamination=0.02, random_state=RANDOM_SEED)
    preds  = iso.fit_predict(y_s)
    results["Isolation Forest"] = int((preds == -1).sum())

    iso_idx = idx[preds == -1]
    df.loc[iso_idx, ["ds", TARGET]].to_csv(
        OUTPUT_DIR / "anomalies_isolation_forest.csv", index=False)

    print("[anomalies]", results)
    return results


# STEP 8 — Metrics from CV

def metrics_from_cv(cv_df: pd.DataFrame, models: list) -> pd.DataFrame:
    rows = []
    y_true = cv_df["y"].values
    for m in models:
        if m not in cv_df.columns:
            continue
        y_pred = cv_df[m].values
        mask   = ~np.isnan(y_pred)
        mae    = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        rmse   = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
        mape   = np.mean(np.abs((y_true[mask] - y_pred[mask]) /
                                (np.abs(y_true[mask]) + 1e-8))) * 100
        rows.append({"model": m,
                     "MAE":   round(mae,  4),
                     "RMSE":  round(rmse, 4),
                     "MAPE%": round(mape, 4)})
    return pd.DataFrame(rows).sort_values("MAE")


# MAIN

def main():
    print("=" * 60)
    print("Brisbane Water Quality — Time Series Pipeline")
    print("=" * 60)

    df = load_and_prepare(DATA_RAW)
    df.to_csv(DATA_CLEAN, index=False)

    df_feat = add_features(df)
    train, test = train_test_split_ts(df, HORIZON)
    anomalies = detect_anomalies(df)

    print("\n[statistical] Running statsforecast models...")
    stat_fcst, stat_cv, sf = run_statistical_models(train, test)
    stat_fcst.to_csv(OUTPUT_DIR / "stat_forecast.csv", index=False)
    stat_cv.to_csv(OUTPUT_DIR   / "stat_cv.csv",       index=False)
    stat_metrics = metrics_from_cv(
        stat_cv, ["SeasonalNaive","Naive","AutoARIMA","AutoETS","AutoTheta","MSTL"])
    stat_metrics.to_csv(OUTPUT_DIR / "stat_metrics.csv", index=False)
    print(stat_metrics.to_string(index=False))

    print("\n[ml] Running mlforecast models...")
    ml_fcst, ml_cv, fcst = run_ml_models(df_feat)
    ml_fcst.to_csv(OUTPUT_DIR / "ml_forecast.csv", index=False)
    ml_cv.to_csv(OUTPUT_DIR   / "ml_cv.csv",       index=False)
    ml_metrics = metrics_from_cv(
        ml_cv, ["LGBMRegressor","RandomForestRegressor","XGBRegressor"])
    ml_metrics.to_csv(OUTPUT_DIR / "ml_metrics.csv", index=False)
    print(ml_metrics.to_string(index=False))

    print("\n[dl] Running neuralforecast models...")
    dl_fcst, _ = run_dl_models(train)
    dl_fcst.to_csv(OUTPUT_DIR / "dl_forecast.csv", index=False)

    print("\n[done] Pipeline complete. Check output/ directory.")


if __name__ == "__main__":
    main()
