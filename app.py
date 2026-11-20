import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

import rag

log = logging.getLogger("rag")
CORS = {"Access-Control-Allow-Origin": "*"}
ABSTAIN = {"answer": None, "sources": [], "scores": [], "confidence": 0.1}

app = FastAPI(title="KB RAG service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = rag.Engine()  # loads model + precomputes embeddings once at startup
engine.answer("warm up Raft quorum", 3)


@app.get("/")
@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "passages": engine.N, "embedder": engine.emb.kind}, headers=CORS)


@app.post("/query")
async def query(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    q = body.get("question")
    q = q if isinstance(q, str) else ""
    try:
        result = await asyncio.wait_for(run_in_threadpool(engine.answer, q, body.get("top_k", 3)), timeout=7.0)
    except Exception:
        log.exception("query failed")
        result = ABSTAIN
    return JSONResponse(result, headers=CORS)
