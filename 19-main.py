import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag import Retriever, extractive_answer, groq_answer

BASE = Path(__file__).resolve().parent
app = FastAPI(title="Ask JNTUH")
retriever = Retriever(BASE / "chunks.json")

class Ask(BaseModel):
    q: str

@app.post("/api/ask")
def ask(body: Ask):
    q = body.q.strip()
    if not q:
        return {"answer": None, "points": [], "sources": [], "mode": "none"}
    hits = retriever.search(q)
    gen = groq_answer(q, hits)
    points = [] if gen else extractive_answer(q, hits, retriever)
    seen, sources = set(), []
    for h, s in hits[:4]:
        if h["doc"] not in seen:
            seen.add(h["doc"])
            sources.append({"doc": h["doc"], "page": h["page"], "url": h["url"], "score": s})
    return {"answer": gen, "points": points, "sources": sources, "mode": "generative" if gen else "extractive"}

@app.get("/api/health")
def health():
    return {"ok": True, "chunks": retriever.N}

app.mount("/static", StaticFiles(directory=BASE), name="static")

@app.get("/")
def home():
    return FileResponse(BASE / "index.html")
