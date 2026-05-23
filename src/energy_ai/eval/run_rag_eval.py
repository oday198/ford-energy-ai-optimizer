import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

# ragas 0.4.3 metrics (warnings are OK)
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from energy_ai.common.paths import ROOT
from energy_ai.agents.genai_analyst import ask_debug

load_dotenv()

OUT_FILE = ROOT / "artifacts" / "reports" / "ragas_eval.json"


class EmbeddingsAdapter:
    def __init__(self, emb):
        self.emb = emb

    def embed_documents(self, texts):
        return self.emb.embed_documents(texts)

    def embed_query(self, text):
        if hasattr(self.emb, "embed_query"):
            return self.emb.embed_query(text)
        return self.emb.embed_documents([text])[0]


def main():
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    questions = [
        {
            "question": "What assumptions does this prototype make about load and price data?",
            "ground_truth": "Load is synthetic mapped from public grid load; price is synthetic correlated with load + daily seasonality + noise.",
        },
        {
            "question": "List the key BESS constraints used in the optimizer.",
            "ground_truth": "Power limit, SOC bounds, SOC dynamics with efficiencies, no simultaneous charge/discharge, no-export grid>=0.",
        },
        {
            "question": "Is this using Ford internal data?",
            "ground_truth": "No, it uses public/open data and synthetic mappings; it is a prototype/reference architecture.",
        },
    ]

    rows = []
    for q in questions:
        dbg = ask_debug(q["question"])
        rows.append(
            {
                "question": q["question"],
                "answer": dbg["answer_text"],
                "contexts": dbg["contexts"],
                "ground_truth": q["ground_truth"],
            }
        )

    ds = Dataset.from_list(rows)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=4096)
    embeddings = EmbeddingsAdapter(OpenAIEmbeddings(model="text-embedding-3-small"))

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()

    # compute mean only for columns that can be coerced to numeric
    base_cols = {"question", "answer", "contexts", "ground_truth"}
    metrics_mean = {}
    for c in df.columns:
        if c in base_cols:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            metrics_mean[c] = float(s.mean())

    payload = {
        "metrics_mean": metrics_mean,
        "rows_scored": df.to_dict(orient="records"),
    }

    OUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved:", OUT_FILE)
    print(json.dumps(metrics_mean, indent=2))


if __name__ == "__main__":
    main()