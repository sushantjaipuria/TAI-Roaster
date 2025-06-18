"""
FastAPI Main Application Module

This file contains:
- FastAPI app initialization
- CORS configuration for frontend integration
- API route registration
- Middleware setup
- Application lifecycle events (startup/shutdown)

Key responsibilities:
- Configure FastAPI app with proper settings
- Enable CORS for Next.js frontend communication
- Register all API routes from routes/ directory
- Set up logging and monitoring
- Handle application startup/shutdown procedures
- Configure middleware for authentication, logging, etc.

Usage:
- Run with: uvicorn backend.app.main:app --reload
- Accessed by frontend at configured API_URL
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
import uvicorn
from contextlib import asynccontextmanager
import logging

from app.api.api import api_router
from app.core.config import settings




@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 TAI Roaster Backend Starting...")
    yield
    # Shutdown
    print("🛑 TAI Roaster Backend Shutting down...")


app = FastAPI(
    title="TAI Roaster API",
    description="Portfolio Analysis and Investment Recommendations API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API router
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "TAI Roaster API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "tai-roaster-api"}


# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail, "message": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "message": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
