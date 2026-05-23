import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from energy_ai.models.features import make_time_features, add_lags, add_rolling
from energy_ai.common.paths import PROCESSED_DIR, ROOT


DATA_FILE = PROCESSED_DIR / "dc_load_price.parquet"
MODEL_OUT = ROOT / "artifacts" / "models" / "load_forecast_xgb.json"
REPORT_OUT = ROOT / "artifacts" / "reports" / "load_forecast_metrics.json"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def main():
    Path(MODEL_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_OUT).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values("ts")

    # Features
    df = make_time_features(df, ts_col="ts")
    df = add_lags(df, target_col="dc_load_mw", lags=[1, 2, 24, 48, 168])
    df = add_rolling(df, target_col="dc_load_mw", windows=[24, 168])

    # Drop rows with NaNs from lag/rolling
    df = df.dropna().reset_index(drop=True)

    target = "dc_load_mw"

    feature_cols = [
        "price_usd_per_mwh",
        "hour", "dow", "month", "is_weekend",
        "dc_load_mw_lag_1h", "dc_load_mw_lag_2h",
        "dc_load_mw_lag_24h", "dc_load_mw_lag_48h", "dc_load_mw_lag_168h",
        "dc_load_mw_rollmean_24h", "dc_load_mw_rollmean_168h",
    ]

    # Time split: last 20% as test
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=7,
        n_jobs=4,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "mae_mw": float(mean_absolute_error(y_test, pred)),
        "rmse_mw": rmse(y_test, pred),
        "mape_pct": mape(y_test, pred),
        "features": feature_cols,
        "model_out": str(MODEL_OUT),
    }

    # Save model (XGBoost native JSON)
    model.save_model(str(MODEL_OUT))

    # Save report
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model:", MODEL_OUT)
    print("Saved report:", REPORT_OUT)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()