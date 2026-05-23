from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
import pulp


@dataclass
class BESSParams:
    capacity_mwh: float = 100.0
    p_max_mw: float = 50.0
    soc_min: float = 0.10
    soc_max: float = 0.90
    eta_c: float = 0.95
    eta_d: float = 0.95
    soc0: float = 0.50          # fraction of capacity
    dt_hours: float = 1.0
    no_export: bool = True


def optimize_bess_dispatch(
    df: pd.DataFrame,
    load_col: str = "dc_load_mw",
    price_col: str = "price_usd_per_mwh",
    params: BESSParams = BESSParams(),
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    d = df.copy()
    d = d.sort_values("ts").reset_index(drop=True)

    n = len(d)
    load = d[load_col].astype(float).tolist()
    price = d[price_col].astype(float).tolist()

    cap = params.capacity_mwh
    pmax = params.p_max_mw
    soc_min_mwh = params.soc_min * cap
    soc_max_mwh = params.soc_max * cap
    soc_init_mwh = params.soc0 * cap
    dt = params.dt_hours

    prob = pulp.LpProblem("bess_dispatch", pulp.LpMinimize)

    ch = pulp.LpVariable.dicts("ch_mw", range(n), lowBound=0, upBound=pmax, cat="Continuous")
    dis = pulp.LpVariable.dicts("dis_mw", range(n), lowBound=0, upBound=pmax, cat="Continuous")
    z = pulp.LpVariable.dicts("is_charging", range(n), lowBound=0, upBound=1, cat="Binary")

    soc = pulp.LpVariable.dicts("soc_mwh", range(n + 1), lowBound=soc_min_mwh, upBound=soc_max_mwh, cat="Continuous")
    grid = pulp.LpVariable.dicts("grid_mw", range(n), lowBound=0, cat="Continuous")

    # Initial SOC
    prob += soc[0] == soc_init_mwh

    for t in range(n):
        # Prevent simultaneous charge/discharge (binary switch)
        prob += ch[t] <= pmax * z[t]
        prob += dis[t] <= pmax * (1 - z[t])

        # Grid import = load + charge - discharge
        prob += grid[t] == load[t] + ch[t] - dis[t]

        if params.no_export:
            prob += grid[t] >= 0

        # SOC dynamics
        prob += soc[t + 1] == soc[t] + (params.eta_c * ch[t] * dt) - ((1.0 / params.eta_d) * dis[t] * dt)

    # Objective: minimize energy cost
    prob += pulp.lpSum([price[t] * grid[t] * dt for t in range(n)])

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    status_name = pulp.LpStatus.get(status, str(status))

    # Build results
    d["bess_charge_mw"] = [pulp.value(ch[t]) for t in range(n)]
    d["bess_discharge_mw"] = [pulp.value(dis[t]) for t in range(n)]
    d["grid_mw_after"] = [pulp.value(grid[t]) for t in range(n)]
    d["soc_mwh"] = [pulp.value(soc[t]) for t in range(n)]
    d["soc_mwh_next"] = [pulp.value(soc[t + 1]) for t in range(n)]

    d["cost_before"] = d[price_col] * d[load_col] * dt
    d["cost_after"] = d[price_col] * d["grid_mw_after"] * dt

    summary = {
        "status": status_name,
        "n_rows": n,
        "cost_before_usd": float(d["cost_before"].sum()),
        "cost_after_usd": float(d["cost_after"].sum()),
        "savings_usd": float(d["cost_before"].sum() - d["cost_after"].sum()),
        "savings_pct": float((d["cost_before"].sum() - d["cost_after"].sum()) / max(d["cost_before"].sum(), 1e-9) * 100.0),
        "params": params.__dict__,
    }
    return d, summary