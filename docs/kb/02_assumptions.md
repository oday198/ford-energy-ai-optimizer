# Assumptions (Prototype)

- Load is a synthetic data-center-like profile mapped from a public grid load series.
- Electricity price is synthetic but correlated with load + daily seasonality + noise.
- BESS model: 1-hour timestep, charge/discharge efficiency, SOC bounds, no-export constraint.
- Optimizer objective: minimize energy purchase cost (no demand charges modeled).
- Results show cost savings for demonstration; production requires real tariff + constraints.