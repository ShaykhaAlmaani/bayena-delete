"""
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.data.catalog import initialize_all_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Bayena Dashboard API starting up — pre-generating sample datasets...")
    initialize_all_datasets()
    logger.info("All datasets ready.")
    yield
    logger.info("Bayena Dashboard API shutting down.")


app = FastAPI(
    title="Bayena Dashboard API",
    description="AI-Powered Environmental Dashboard Generation for the Bayena Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "service": "Bayena / بيّنة Dashboard API",
        "docs": "/docs",
        "health": "/api/health",
    }
