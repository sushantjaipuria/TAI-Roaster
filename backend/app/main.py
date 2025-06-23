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

# Load environment variables first
from dotenv import load_dotenv
import os
load_dotenv()  # Load .env file from project root

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
import uvicorn
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

from app.api.api import api_router
from app.core.config import settings

# Add project root to Python path for intelligence module
project_root = Path(__file__).parent.parent.parent  # Go up to TAI-Roaster root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    print(f"✅ Added project root to Python path: {project_root}")

# Try to import intelligence components
try:
    from app.core.ml_models import model_manager
    from app.services.intelligence_service import intelligence_service
    from app.services.market_data_service import market_data_service
    INTELLIGENCE_MODULE_AVAILABLE = True
    print("✅ Intelligence module imports successful")
except ImportError as e:
    print(f"⚠️  Intelligence module not available: {e}")
    INTELLIGENCE_MODULE_AVAILABLE = False




@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 TAI Roaster Backend Starting...")
    
    # Initialize intelligence components if available
    if INTELLIGENCE_MODULE_AVAILABLE:
        try:
            print("🤖 Loading ML models...")
            
            # Models are loaded automatically when model_manager is imported
            model_info = model_manager.get_model_info()
            print(f"✅ ML Models Status: {model_info['total_models']} models")
            
            # Initialize services
            if intelligence_service.initialized:
                print("✅ Intelligence Service initialized")
            else:
                print("⚠️  Intelligence Service initialization failed")
                
            if market_data_service.initialized:
                print("✅ Market Data Service initialized")
            else:
                print("⚠️  Market Data Service initialization failed")
                
            # Store in app state for access in endpoints
            app.state.model_manager = model_manager
            app.state.intelligence_service = intelligence_service
            app.state.market_data_service = market_data_service
            
        except Exception as e:
            print(f"❌ Error initializing intelligence components: {e}")
    else:
        print("⚠️  Intelligence module not available - running in basic mode")
    
    print("✅ TAI Roaster Backend Ready!")
    
    yield
    
    # Shutdown
    print("🛑 TAI Roaster Backend Shutting down...")
    
    # Cleanup if needed
    if INTELLIGENCE_MODULE_AVAILABLE:
        try:
            if hasattr(market_data_service, 'clear_cache'):
                market_data_service.clear_cache()
                print("✅ Market data cache cleared")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")


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
