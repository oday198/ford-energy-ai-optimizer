import json
from pathlib import Path

import pandas as pd

from energy_ai.common.paths import PROCESSED_DIR, ROOT
from energy_ai.optimizer.dispatch_milp import BESSParams, optimize_bess_dispatch


DATA_FILE = PROCESSED_DIR / "dc_load_price.parquet"
OUT_TS = ROOT / "artifacts" / "reports" / "dispatch_timeseries.parquet"
OUT_SUMMARY = ROOT / "artifacts" / "reports" / "dispatch_summary.json"


def main():
    Path(OUT_TS).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_FILE).sort_values("ts")

    # Use a limited horizon so it solves quickly on a laptop (change if you want)
    horizon_hours = 24 * 14  # 14 days
    df_h = df.iloc[:horizon_hours].copy()

    params = BESSParams(
        capacity_mwh=100.0,
        p_max_mw=50.0,
        soc_min=0.10,
        soc_max=0.90,
        eta_c=0.95,
        eta_d=0.95,
        soc0=0.50,
        dt_hours=1.0,
        no_export=True,
    )

    ts, summary = optimize_bess_dispatch(df_h, params=params)

    ts.to_parquet(OUT_TS, index=False)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", OUT_TS)
    print("Saved:", OUT_SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()