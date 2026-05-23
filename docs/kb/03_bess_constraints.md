# BESS constraints used in the optimizer

- Power limit Pmax (MW) for charging/discharging
- SOC bounds: SOC_min..SOC_max
- SOC dynamics: SOC(t+1) = SOC(t) + eta_c*charge - discharge/eta_d
- No simultaneous charge/discharge (binary switch)
- No export: grid import >= 0