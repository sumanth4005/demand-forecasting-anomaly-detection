"""Run the full pipeline: python run.py [--csv path/to/daily_demand.csv]"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from src.forecast import (load_data, make_features, time_split, train_and_compare,
                          flag_anomalies, drift_check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with columns date,demand (default: built-in synthetic series)")
    ap.add_argument("--test-days", type=int, default=120)
    args = ap.parse_args()

    if args.csv:
        raw, spikes = load_data(args.csv), None
    else:
        raw, spikes = load_data(return_spikes=True)
    df = make_features(raw)
    train, test = time_split(df, args.test_days)
    results, best, model, features = train_and_compare(train, test)

    print("Test RMSE by model (lower is better):")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {k:<24} {v:7.2f}")
    base = results["Seasonal naive (lag_7)"]
    print(f"\nBest model: {best} ({(base - results[best]) / base:.0%} lower RMSE than the seasonal-naive baseline)")

    scored = flag_anomalies(model, features, train, test)
    print(f"MAE on test window: {mean_absolute_error(scored['demand'], scored['forecast']):.2f}")
    flagged = scored.index[scored["anomaly"]]
    print(f"Anomalies flagged: {len(flagged)}")
    if spikes is not None:
        truth = spikes[(spikes >= test.index.min()) & (spikes <= test.index.max())]
        hits = len(set(flagged) & set(truth))
        print(f"Injected anomalies in test window: {len(truth)} | caught: {hits} | "
              f"false alarms: {len(flagged) - hits}")

    change, drifted = drift_check(train, test)
    print(f"Drift check: {change:+.1%} change in 28-day mean -> retrain needed: {drifted}")

    os.makedirs("docs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(scored.index, scored["demand"], label="actual", lw=1.2)
    ax.plot(scored.index, scored["forecast"], label=f"forecast ({best})", lw=1.2)
    ax.scatter(flagged, scored.loc[flagged, "demand"], color="red", zorder=3, label="flagged anomaly")
    ax.set_title("Test window: actual vs forecast, with flagged anomalies")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("docs/forecast_test_window.png", dpi=120)
    print("Saved chart to docs/forecast_test_window.png")


if __name__ == "__main__":
    main()
