"""
ECHO — FastAPI Backend
All endpoints used by the Next.js frontend.
Docs: http://localhost:8000/docs

Run from the project root with:
  uvicorn backend.main:app --reload --port 8000
"""

import json
import asyncio
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml.inference import engine   # clean package import


# ── Startup / Shutdown ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run blocking model load in a thread pool so we don't block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.initialize)
    yield
    # Nothing to clean up


app = FastAPI(
    title       = "ECHO API",
    description = "Explainable Cognitive Denoising for Personalized Recommendations",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Allow Next.js dev server (port 3000) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Helper ────────────────────────────────────────────────────────────────
async def run_blocking(fn, *args, **kwargs):
    """Run a blocking (CPU/GPU) function in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ECHO API is running 🚀",
            "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model_loaded": engine._initialized}


# ── Users ─────────────────────────────────────────────────────────────────

@app.get("/users", tags=["Users"])
async def list_users(
    limit:  int = Query(50,  ge=1, le=500,  description="Max users to return"),
    offset: int = Query(0,   ge=0,           description="Pagination offset"),
):
    """
    List demo users from the dataset.
    Returns user IDs and a preview of their history length.
    """
    all_ids  = engine.get_user_ids()
    page     = all_ids[offset: offset + limit]
    result   = []
    for uid in page:
        history = engine.get_user_history(uid)
        result.append({
            "user_id"        : uid,
            "interaction_count": len(history) - 1,   # exclude test item
            "display_name"   : f"User {uid.replace('user', '#')}",
        })
    return {"total": len(all_ids), "users": result}


@app.get("/users/{user_id}/history", tags=["Users"])
async def get_user_history(user_id: str):
    """
    Full user interaction history with per-item noise detection.
    Each item has: kept/noised label, c1/c2/c3 scores, explanation.
    """
    profile = await run_blocking(engine.get_user_profile, user_id)
    if profile is None:
        raise HTTPException(status_code=404,
                            detail=f"User '{user_id}' not found or has insufficient data.")
    return profile


@app.get("/users/{user_id}/recommend", tags=["Recommendations"])
async def get_recommendations(
    user_id: str,
    k: int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    """
    Top-K recommendations for a user after denoising their history.
    """
    recs = await run_blocking(engine.get_recommendations, user_id, k)
    if recs is None:
        raise HTTPException(status_code=404,
                            detail=f"User '{user_id}' not found or has no semantic data.")
    return {"user_id": user_id, "recommendations": recs, "k": k}


@app.get("/metrics", tags=["Analytics"])
async def get_metrics():
    """
    Evaluation metrics: HR@K and NDCG@K for IADSR+ vs baselines.
    Used by the analytics dashboard.
    """
    metrics = engine.get_metrics()
    if not metrics:
        return {"message": "Run ml/evaluate.py first to generate metrics.",
                "metrics": {}}
    return metrics


@app.get("/items/{item_id}", tags=["Items"])
async def get_item(item_id: int):
    """Look up an item by ID."""
    title = engine.get_item_title(item_id)
    return {"item_id": item_id, "title": title}


@app.get("/stats", tags=["Analytics"])
async def get_stats():
    """High-level dataset statistics for the dashboard."""
    all_ids   = engine.get_user_ids()
    lengths   = [len(engine.get_user_history(uid)) - 1 for uid in all_ids[:500]]
    return {
        "total_users"  : len(all_ids),
        "total_items"  : 12102,
        "dataset"      : "Amazon Beauty",
        "avg_seq_len"  : round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "encoder"      : "BAAI/bge-large-en-v1.5 (768-dim)",
        "backbone"     : "GRU4Rec",
        "novel_contrib": [
            "Attention-weighted cross-modal noise scoring",
            "Lightweight BGE-large encoder (24x smaller than Llama-3.1-8B)",
        ],
    }
