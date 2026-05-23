import json

import pandas as pd
import streamlit as st

from energy_ai.common.paths import ROOT
from energy_ai.agents.genai_analyst import ask


SUMMARY_FILE = ROOT / "artifacts" / "reports" / "dispatch_summary.json"
TS_FILE = ROOT / "artifacts" / "reports" / "dispatch_timeseries.parquet"

st.set_page_config(page_title="Ford Energy AI Optimizer", layout="wide")


@st.cache_data
def load_summary():
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_ts():
    df = pd.read_parquet(TS_FILE)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


def money(x: float) -> str:
    return f"${x:,.0f}"


def main():
    st.title("Ford Energy AI Optimizer — BESS Dispatch Intelligence")
    st.caption("Forecasting + optimization baseline (MILP) + RAG agent + guardrails + Langfuse tracing.")

    if (not SUMMARY_FILE.exists()) or (not TS_FILE.exists()):
        st.error("Missing artifacts. Run: python -m energy_ai.optimizer.run_dispatch")
        st.stop()

    summary = load_summary()
    df = load_ts()

    # ---- KPI row ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Solver status", summary["status"])
    c2.metric("Horizon (hours)", summary["n_rows"])
    c3.metric("Cost before", money(summary["cost_before_usd"]))
    c4.metric("Savings", f'{money(summary["savings_usd"])}  ({summary["savings_pct"]:.2f}%)')

    st.divider()

    # ---- GenAI Analyst ----
    st.subheader("GenAI Analyst (RAG + Agentic Workflow)")
    with st.container(border=True):
        q = st.text_input(
            "Ask about anomalies, drivers of savings, risky hours, assumptions, constraints, methodology",
            value="What assumptions and constraints does this optimizer use? Provide manager-ready summary.",
        )

        colA, colB = st.columns([1, 4])
        run = colA.button("Analyze", type="primary")
        colB.caption("Uses local artifacts + KB docs; does not claim Ford internal data. Traced in Langfuse.")

        if run:
            with st.spinner("Thinking..."):
                try:
                    resp = ask(q)

                    st.markdown("**Executive summary**")
                    st.write(resp.executive_summary)

                    st.markdown("**Key findings**")
                    for x in resp.key_findings:
                        st.write(f"- {x}")

                    st.markdown("**Detected anomalies**")
                    if resp.detected_anomalies:
                        for x in resp.detected_anomalies:
                            st.write(f"- {x}")
                    else:
                        st.write("None detected.")

                    st.markdown("**Recommendations**")
                    if resp.recommendations:
                        for x in resp.recommendations:
                            st.write(f"- {x}")
                    else:
                        st.write("No recommendations returned.")

                    with st.expander("Assumptions"):
                        for x in resp.assumptions:
                            st.write(f"- {x}")

                    with st.expander("RAG Sources (KB files used)"):
                        if getattr(resp, "sources", None):
                            for s in resp.sources:
                                st.write(f"- {s}")
                        else:
                            st.write("No sources.")

                except Exception as e:
                    st.error(str(e))

    st.divider()

    # ---- Charts ----
    st.subheader("Time Series")
    left, right = st.columns(2)

    with left:
        st.markdown("**Load & Grid After Dispatch (MW)**")
        chart_df = df.set_index("ts")[["dc_load_mw", "grid_mw_after"]]
        st.line_chart(chart_df)

    with right:
        st.markdown("**BESS Power (MW)**")
        p = df.copy()
        p["bess_net_mw"] = p["bess_discharge_mw"] - p["bess_charge_mw"]
        st.line_chart(p.set_index("ts")[["bess_charge_mw", "bess_discharge_mw", "bess_net_mw"]])

    st.divider()

    left2, right2 = st.columns(2)

    with left2:
        st.markdown("**State of Charge (MWh)**")
        st.line_chart(df.set_index("ts")[["soc_mwh", "soc_mwh_next"]])

    with right2:
        st.markdown("**Electricity Price ($/MWh)**")
        st.line_chart(df.set_index("ts")[["price_usd_per_mwh"]])

    st.divider()

    st.subheader("Cost Breakdown")
    cost = df.copy()
    cost = cost.set_index("ts")[["cost_before", "cost_after"]]
    st.area_chart(cost)

    with st.expander("Parameters"):
        st.json(summary["params"])

    with st.expander("Sample table"):
        st.dataframe(df.head(25))


if __name__ == "__main__":
    main()