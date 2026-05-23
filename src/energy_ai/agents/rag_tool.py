from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from energy_ai.common.paths import ROOT

load_dotenv()

CHROMA_DIR = ROOT / "artifacts" / "chroma_kb"
COLLECTION = "ford_energy_kb"


def _get_vs():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def retrieve_kb(query: str, k: int = 4) -> List[Dict[str, Any]]:
    vs = _get_vs()
    docs = vs.similarity_search(query, k=k)
    return [
        {"source": d.metadata.get("source", "unknown"), "text": d.page_content}
        for d in docs
    ]


def retrieve_kb_multi(queries: List[str], k_per_query: int = 4, max_unique_sources: int = 6) -> List[Dict[str, Any]]:
    vs = _get_vs()

    hits: List[Dict[str, Any]] = []
    for q in queries:
        docs = vs.similarity_search(q, k=k_per_query)
        for d in docs:
            hits.append({"source": d.metadata.get("source", "unknown"), "text": d.page_content})

    # dedupe by source (keep first)
    seen = set()
    uniq = []
    for h in hits:
        s = h.get("source", "")
        if s and s not in seen:
            uniq.append(h)
            seen.add(s)
        if len(uniq) >= max_unique_sources:
            break

    return uniq