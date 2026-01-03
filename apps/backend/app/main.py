from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api import cases, documents
from shared.database import init_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("[backend] LexIntel backend starting...")
    await init_db()
    logger.info("[backend] Database initialized")
    yield
    # Shutdown
    logger.info("[backend] LexIntel backend shutting down...")

app = FastAPI(
    title="LexIntel API",
    description="AI-powered legal research platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "LexIntel API v0.1.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Include routers
app.include_router(cases.router)
app.include_router(documents.router)

# TODO: Add more routers
# from app.api import search, chat
# app.include_router(search.router)
# app.include_router(chat.router)
