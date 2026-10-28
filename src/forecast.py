"""
Demand Forecasting & Anomaly Detection - hands-on walkthrough
Python: pandas, NumPy, scikit-learn

Steps
 1. Load and clean daily demand data
 2. Feature engineering: lags, rolling windows, calendar
 3. Time-based train/test split (no future leakage)
 4. Baseline vs Linear Regression vs Random Forest vs Gradient Boosting (compare RMSE)
 5. Anomaly detection from forecast residuals
 6. Simple drift check
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------------------------------------------------------------- 1. Data
def load_data(path=None, return_spikes=False):
    """Read a CSV with columns: date, demand. Falls back to a synthetic series for the demo.

    With return_spikes=True (synthetic data only) also returns the dates of the injected anomalies,
    so the anomaly detector can be scored against known ground truth.
    """
    spike_dates = pd.DatetimeIndex([])
    if path:
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", "2025-06-30", freq="D")
        t = np.arange(len(dates))
        demand = (200 + 0.08 * t                                   # trend
                  + 25 * np.sin(2 * np.pi * dates.dayofweek / 7)   # weekly cycle
                  + 40 * np.sin(2 * np.pi * dates.dayofyear / 365) # yearly cycle
                  + rng.normal(0, 10, len(dates))).to_numpy().copy()
        spikes = rng.choice(len(dates), 12, replace=False)         # injected anomalies
        demand[spikes] += rng.choice([-1, 1], 12) * rng.uniform(80, 140, 12)
        spike_dates = dates[np.sort(spikes)]
        df = pd.DataFrame({"date": dates, "demand": demand})

    df = (df.sort_values("date")
            .drop_duplicates("date")
            .set_index("date")
            .asfreq("D"))
    df["demand"] = df["demand"].interpolate(limit=3)   # fill short gaps only
    df = df.dropna()
    return (df, spike_dates) if return_spikes else df


# ---------------------------------------------------------------- 2. Features
def make_features(df):
    out = df.copy()
    for lag in [1, 7, 14, 28]:
        out[f"lag_{lag}"] = out["demand"].shift(lag)
    # shift(1) before rolling so a row never sees its own value
    s = out["demand"].shift(1)
    for w in [7, 28]:
        out[f"roll_mean_{w}"] = s.rolling(w).mean()
        out[f"roll_std_{w}"] = s.rolling(w).std()
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out.dropna()


# ---------------------------------------------------------------- 3. Split
def time_split(df, test_days=120):
    cutoff = df.index.max() - pd.Timedelta(days=test_days)
    return df[df.index <= cutoff], df[df.index > cutoff]


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# ---------------------------------------------------------------- 4. Models
def train_and_compare(train, test):
    features = [c for c in train.columns if c != "demand"]
    X_tr, y_tr = train[features], train["demand"]
    X_te, y_te = test[features], test["demand"]

    results = {"Seasonal naive (lag_7)": rmse(y_te, X_te["lag_7"])}
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
    }
    fitted = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        results[name] = rmse(y_te, m.predict(X_te))
        fitted[name] = m

    best = min(fitted, key=lambda n: results[n])
    return results, best, fitted[best], features


# ---------------------------------------------------------------- 5. Anomalies
def flag_anomalies(model, features, train, test, z=3.0):
    """Flag days whose forecast residual is more than z standard deviations from normal."""
    resid_train = train["demand"] - model.predict(train[features])
    mu, sd = resid_train.mean(), resid_train.std()
    out = test[["demand"]].copy()
    out["forecast"] = model.predict(test[features])
    out["residual"] = out["demand"] - out["forecast"]
    out["z"] = (out["residual"] - mu) / sd
    out["anomaly"] = out["z"].abs() > z
    return out


# ---------------------------------------------------------------- 6. Drift
def drift_check(train, test, col="roll_mean_28", tol=0.15):
    """Flag drift if the recent mean of a key feature moved more than tol (15%)."""
    base, recent = train[col].tail(90).mean(), test[col].tail(30).mean()
    change = (recent - base) / base
    return change, abs(change) > tol


def predict_next(model, features, row):
    """API-ready inference: one feature row (dict) in, one forecast out."""
    return float(model.predict(pd.DataFrame([row])[features])[0])


