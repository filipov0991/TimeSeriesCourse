
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

DATA_CLEAN = Path("data/prepared_temperature_ts.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET     = "Temperature"
SEASON_LEN = 48


def stationarity_tests(series: pd.Series, name: str):
    print(f"\n=== Stationarity Tests: {name} ===")

    adf = adfuller(series.dropna(), autolag="AIC")
    print(f"ADF  stat={adf[0]:.4f}  p={adf[1]:.4f}  => "
          f"{'Stationary' if adf[1] < 0.05 else 'Non-stationary'}")

    kpss_res = kpss(series.dropna(), regression="c", nlags="auto")
    print(f"KPSS stat={kpss_res[0]:.4f}  p={kpss_res[1]:.4f}  => "
          f"{'Non-stationary' if kpss_res[1] < 0.05 else 'Stationary'}")


def run_eda():
    df = pd.read_csv(DATA_CLEAN, parse_dates=["ds"])
    df = df.set_index("ds")
    y  = df[TARGET]

    # ── Basic stats ──────────────────────────────────────────────────────────
    print("\n=== Basic Statistics ===")
    print(y.describe().round(3))
    print(f"Missing values: {y.isna().sum()}")
    print(f"Period: {y.index.min()} — {y.index.max()}")
    print(f"Total observations: {len(y)}")

    # ── Stationarity ─────────────────────────────────────────────────────────
    stationarity_tests(y, TARGET)
    y_diff = y.diff().dropna()
    stationarity_tests(y_diff, f"{TARGET} (1st diff)")

    # ── Seasonal decomposition ───────────────────────────────────────────────
    decomp = seasonal_decompose(y.dropna(), model="additive", period=SEASON_LEN)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(decomp.observed,  color="#1f77b4", lw=0.8);  axes[0].set_ylabel("Observed")
    axes[1].plot(decomp.trend,     color="#ff7f0e", lw=1.2);  axes[1].set_ylabel("Trend")
    axes[2].plot(decomp.seasonal,  color="#2ca02c", lw=0.8);  axes[2].set_ylabel("Seasonal")
    axes[3].plot(decomp.resid,     color="#d62728", lw=0.6, alpha=0.7);  axes[3].set_ylabel("Residual")
    fig.suptitle(f"Seasonal Decomposition — {TARGET} (period={SEASON_LEN})", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save] output/eda_decomposition.png")

    # ── ACF / PACF ───────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    plot_acf(y.dropna(),  lags=96, ax=ax1, title=f"ACF — {TARGET}")
    plot_pacf(y.dropna(), lags=96, ax=ax2, title=f"PACF — {TARGET}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[save] output/eda_acf_pacf.png")

    # ── Rolling stats (stationarity visual) ──────────────────────────────────
    roll = y.rolling(SEASON_LEN)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y,               label="Original",         lw=0.7, alpha=0.7)
    ax.plot(roll.mean(),     label="Rolling Mean (48)", lw=1.5)
    ax.plot(roll.std(),      label="Rolling Std (48)",  lw=1.2, linestyle="--")
    ax.set_title(f"Rolling Statistics — {TARGET}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_rolling_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[save] output/eda_rolling_stats.png")

    # ── Correlation heatmap ──────────────────────────────────────────────────
    import seaborn as sns
    num_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(num_df.corr().round(2), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[save] output/eda_correlation.png")

    # ── Hourly boxplot ───────────────────────────────────────────────────────
    df_h = df.copy()
    df_h["hour"] = df_h.index.hour
    fig, ax = plt.subplots(figsize=(14, 4))
    df_h.boxplot(column=TARGET, by="hour", ax=ax, grid=False)
    ax.set_title(f"Hourly Distribution — {TARGET}")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel(TARGET)
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_hourly_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[save] output/eda_hourly_boxplot.png")

    print("\n[EDA done]")


if __name__ == "__main__":
    run_eda()
