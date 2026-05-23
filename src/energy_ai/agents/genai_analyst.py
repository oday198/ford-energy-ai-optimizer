import json
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from energy_ai.common.paths import ROOT
from energy_ai.common.observability import get_langfuse_client
from energy_ai.agents.rag_tool import retrieve_kb_multi

load_dotenv()

SUMMARY_FILE = ROOT / "artifacts" / "reports" / "dispatch_summary.json"
TS_FILE = ROOT / "artifacts" / "reports" / "dispatch_timeseries.parquet"


def validate_question(q: str) -> str:
    q = (q or "").strip()
    if len(q) == 0:
        raise ValueError("Empty question.")
    if len(q) > 600:
        q = q[:600]
    banned = ["password", "api key", "secret", "credit card", "ssn"]
    if any(b in q.lower() for b in banned):
        raise ValueError("Blocked by policy.")
    return q


def load_summary() -> Dict[str, Any]:
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ts() -> pd.DataFrame:
    df = pd.read_parquet(TS_FILE)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


def compute_insights(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["horizon_hours"] = int(len(df))
    out["price_median"] = float(df["price_usd_per_mwh"].median())
    out["price_p95"] = float(df["price_usd_per_mwh"].quantile(0.95))

    top_price = (
        df[["ts", "price_usd_per_mwh"]]
        .sort_values("price_usd_per_mwh", ascending=False)
        .head(10)
        .assign(ts=lambda x: x["ts"].astype(str))
        .to_dict(orient="records")
    )
    out["top_price_hours"] = top_price

    df2 = df.copy()
    df2["savings_usd"] = df2["cost_before"] - df2["cost_after"]
    top_save = (
        df2[["ts", "savings_usd", "price_usd_per_mwh"]]
        .sort_values("savings_usd", ascending=False)
        .head(10)
        .assign(ts=lambda x: x["ts"].astype(str))
        .to_dict(orient="records")
    )
    out["top_savings_hours"] = top_save

    out["soc_min"] = float(df["soc_mwh"].min())
    out["soc_max"] = float(df["soc_mwh"].max())
    return out


class AnalystResponse(BaseModel):
    executive_summary: str = Field(...)
    key_findings: List[str] = Field(...)
    detected_anomalies: List[str] = Field(...)
    recommendations: List[str] = Field(...)
    assumptions: List[str] = Field(...)
    sources: List[str] = Field(default_factory=list)


class RouterDecision(BaseModel):
    action: str = Field(..., description="retrieve or skip")
    search_query: str = Field(default="", description="query to search the KB")


class State(TypedDict):
    question: str
    summary: Dict[str, Any]
    insights: Dict[str, Any]
    kb_hits: List[Dict[str, Any]]
    answer: Optional[AnalystResponse]


def node_load(state: State) -> State:
    state["summary"] = load_summary()
    df = load_ts()
    state["insights"] = compute_insights(df)
    state["kb_hits"] = []
    return state


def node_router(state: State) -> State:
    q = validate_question(state["question"])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(RouterDecision)

    decision = llm.invoke(
        [
            ("system", "Decide whether you need to retrieve from the knowledge base to answer well."),
            ("human", f"Question: {q}\nReturn action='retrieve' when the user asks about assumptions, constraints, methodology, or definitions."),
        ]
    )

    state["kb_hits"] = [{"_router": decision.model_dump()}]
    return state


def node_retrieve(state: State) -> State:
    q = validate_question(state["question"])

    router = state["kb_hits"][0].get("_router", {}) if state.get("kb_hits") else {}
    action = (router.get("action") or "retrieve").lower().strip()
    search_query = (router.get("search_query") or q).strip()

    if action != "retrieve":
        state["kb_hits"] = []
        return state

    queries = [
        search_query,
        "project overview prototype reference architecture",
        "assumptions synthetic load synthetic price",
        "BESS constraints SOC bounds no-export charge discharge efficiency",
    ]
    hits = retrieve_kb_multi(queries, k_per_query=4, max_unique_sources=10)
    state["kb_hits"] = hits
    return state


def node_llm(state: State) -> State:
    q = validate_question(state["question"])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    structured = llm.with_structured_output(AnalystResponse)

    kb = state.get("kb_hits", [])
    kb_text = "\n\n".join(
        [f"SOURCE: {h.get('source')}\n{h.get('text')}" for h in kb if "text" in h]
    )[:6000]

    context = {
        "dispatch_summary": state["summary"],
        "computed_insights": state["insights"],
        "kb_context": kb_text,
        "question": q,
        "response_rules": [
            "Do NOT claim Ford internal data.",
            "Use provided numbers for savings/price/SOC.",
            "If KB context exists, cite sources by listing filenames in sources[].",
        ],
    }

    messages = [
        ("system", "You are an energy analytics copilot for a BESS dispatch dashboard."),
        ("human", f"Context JSON:\n{json.dumps(context, indent=2)}\n\nAnswer the question."),
    ]

    resp = structured.invoke(messages)
    resp.sources = sorted(list({h.get("source", "") for h in kb if h.get("source")}))

    # Langfuse trace
    lf = get_langfuse_client()
    if lf:
        t = lf.trace(name="ford-energy-ai-analyst", input={"question": q})
        t.generation(
            name="rag_answer",
            model="gpt-4o-mini",
            input={"messages": messages, "sources": resp.sources},
            output=resp.model_dump(),
        )
        lf.flush()

    state["answer"] = resp
    return state


def build_agent():
    g = StateGraph(State)
    g.add_node("load", node_load)
    g.add_node("router", node_router)
    g.add_node("retrieve", node_retrieve)
    g.add_node("llm", node_llm)

    g.set_entry_point("load")
    g.add_edge("load", "router")
    g.add_edge("router", "retrieve")
    g.add_edge("retrieve", "llm")
    g.add_edge("llm", END)

    return g.compile()


def ask(question: str) -> AnalystResponse:
    agent = build_agent()
    out = agent.invoke({"question": question})
    return out["answer"]


def ask_debug(question: str) -> Dict[str, Any]:
    """Returns full state including retrieved contexts for evaluation/debug."""
    agent = build_agent()
    out = agent.invoke({"question": question})
    answer: AnalystResponse = out["answer"]
    kb_hits = out.get("kb_hits", [])
    contexts = [h.get("text", "") for h in kb_hits if isinstance(h, dict) and h.get("text")]
    sources = [h.get("source", "") for h in kb_hits if isinstance(h, dict) and h.get("source")]
    return {
        "question": question,
        "answer_text": " ".join(
            [
                answer.executive_summary,
                "\n".join(answer.key_findings),
                "\n".join(answer.assumptions),
                "\n".join(answer.recommendations),
            ]
        ).strip(),
        "contexts": contexts,
        "sources": sorted(list(set(sources))),
        "answer_obj": answer,
    }