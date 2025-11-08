"""FastAPI main application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import settings
from .routers import verify
from .services.tfidf_model import tfidf_service
from .services.distilbert_model import distilbert_service
from .services.nb_model import nb_service


# Create FastAPI app
app = FastAPI(
    title="HDR AI Proposal Verification Assistant",
    description="ML-powered proposal compliance verification system",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event: Load ML models
@app.on_event("startup")
async def load_models():
    """Load all ML models at startup."""
    print("\n" + "=" * 60)
    print("HDR AI Proposal Verification Assistant - Backend Starting")
    print("=" * 60 + "\n")

    print("Loading ML models...")
    tfidf_service.load()
    distilbert_service.load()
    nb_service.load()

    print("\n" + "=" * 60)
    print("Backend ready to accept requests")
    print("=" * 60 + "\n")


# Health check endpoints
@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health():
    """API health check."""
    return {
        "status": "healthy",
        "models": {
            "tfidf": tfidf_service.is_loaded(),
            "distilbert": distilbert_service.is_loaded(),
            "naive_bayes": nb_service.is_loaded(),
        }
    }


# Include API routers
app.include_router(verify.router, prefix="/api")


# Serve React frontend in production (after build)
# This will be mounted at the end so API routes take precedence
try:
    frontend_dist = Path("/app/frontend/dist")
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
        print(f"✓ Serving frontend from {frontend_dist}")
except Exception as e:
    print(f"⚠ Frontend not mounted: {e}")
