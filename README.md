# Ford Energy AI Optimizer — BESS Dispatch Intelligence (Prototype)

End-to-end prototype aligned to Ford GDIA role:
- ML load forecasting (XGBoost)
- BESS dispatch optimization (MILP)
- Agentic GenAI analyst (LangGraph) + RAG (Chroma) + citations
- Guardrails (input validation) + observability (Langfuse tracing)
- RAG evaluation report (RAGAS)

## Local Run (Python)
```bash
pip install -e .
python -m energy_ai.data.make_dataset
python -m energy_ai.optimizer.run_dispatch
python -m energy_ai.data.build_kb
streamlit run src/energy_ai/ui/app.py


API
uvicorn energy_ai.api.main:app --host 0.0.0.0 --port 8000

Docker
docker build -t ford-energy-ai-optimizer:local .
docker compose up

Eval
python -m energy_ai.eval.run_rag_eval