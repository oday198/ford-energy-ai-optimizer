from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from energy_ai.agents.genai_analyst import ask

load_dotenv()

app = FastAPI(title="Ford Energy AI Optimizer API", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=600)


class AskResponse(BaseModel):
    executive_summary: str
    key_findings: list[str]
    detected_anomalies: list[str]
    recommendations: list[str]
    assumptions: list[str]
    sources: list[str] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agent/ask", response_model=AskResponse)
def agent_ask(req: AskRequest):
    resp = ask(req.question)
    return AskResponse(**resp.model_dump())