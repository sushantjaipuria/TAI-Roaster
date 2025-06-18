"""
Portfolio API Endpoints

This module provides API endpoints for portfolio management:
- File upload and parsing
- Portfolio validation
- Sample format information
- Portfolio CRUD operations

Integration:
- Uses new input schemas for validation
- Integrates with file parser service
- Integrates with portfolio validation service
- Provides comprehensive error handling
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import uuid
from datetime import datetime
import asyncio
import logging

from app.schemas.input import (
    PortfolioInput, 
    UserProfile,
    AnalysisRequest,
    FileUploadRequest,
    FileParseResponse,
    PortfolioValidationResponse,
    BulkPortfolioUpload
)
from app.services.file_parser import FileParserService
from app.services.portfolio_validator import PortfolioValidationService
from app.core.config import settings

router = APIRouter()

# Configure logger
logger = logging.getLogger(__name__)

# In-memory storage for demo purposes (in production, use proper database)
portfolios_storage: Dict[str, PortfolioInput] = {}
user_profiles_storage: Dict[str, UserProfile] = {}
validation_cache: Dict[str, PortfolioValidationResponse] = {}

# Service instances
file_parser = FileParserService()
validator = PortfolioValidationService()


@router.post("/upload")
async def upload_portfolio_file(
    file: UploadFile = File(...),
    filename: Optional[str] = Form(None),
    fileSize: Optional[int] = Form(None),
    contentType: Optional[str] = Form(None)
):
    """
    Upload and parse portfolio file (CSV, Excel, TSV).
    
    Returns parsed portfolio data with validation results.
    """
    logger.info(f"📁 Upload request received - filename: {filename}, size: {fileSize}, type: {contentType}")
    
    try:
        # Create file upload request
        file_request = FileUploadRequest(
            filename=filename or file.filename or "unknown.csv",
            file_size=fileSize or 0,
            content_type=contentType or file.content_type
        )
        logger.info(f"📋 FileUploadRequest created: {file_request}")
        
        # Validate file format first
        format_valid, format_errors = await file_parser.validate_file_format(file_request)
        logger.info(f"🔍 Format validation - valid: {format_valid}, errors: {format_errors}")
        
        if not format_valid:
            logger.warning(f"❌ Format validation failed: {format_errors}")
            return {
                "success": True,
                "data": FileParseResponse(
                    success=False,
                    portfolio=None,
                    errors=format_errors,
                    warnings=[],
                    rows_processed=0,
                    rows_skipped=0
                )
            }
        
        # Read file content
        content = await file.read()
        logger.info(f"📖 File content read - length: {len(content)} bytes")
        logger.debug(f"📄 First 200 chars: {content[:200]}")
        
        if len(content) == 0:
            logger.error("📭 File is empty")
            return {
                "success": True,
                "data": FileParseResponse(
                    success=False,
                    portfolio=None,
                    errors=["File is empty"],
                    warnings=[],
                    rows_processed=0,
                    rows_skipped=0
                )
            }
        
        # Parse file
        logger.info("🔄 Starting file parsing...")
        parse_response = await file_parser.parse_file(content, file_request)
        logger.info(f"✅ Parse response - success: {parse_response.success}, rows processed: {parse_response.rows_processed}")
        
        if parse_response.errors:
            logger.warning(f"⚠️ Parse errors: {parse_response.errors}")
        if parse_response.warnings:
            logger.info(f"⚠️ Parse warnings: {parse_response.warnings}")
        
        # Store portfolio if parsing was successful
        if parse_response.success and parse_response.portfolio:
            session_id = str(uuid.uuid4())
            portfolios_storage[session_id] = parse_response.portfolio
            logger.info(f"💾 Portfolio stored with session_id: {session_id}")
        
        return {
            "success": True,
            "data": parse_response
        }
        
    except Exception as e:
        logger.error(f"❌ Upload failed with exception: {str(e)}", exc_info=True)
        return {
            "success": True,
            "data": FileParseResponse(
                success=False,
                portfolio=None,
                errors=[f"Upload failed: {str(e)}"],
                warnings=[],
                rows_processed=0,
                rows_skipped=0
            )
        }


@router.post("/validate", response_model=PortfolioValidationResponse)
async def validate_portfolio(
    request: AnalysisRequest
):
    """
    Validate portfolio data against business rules and user profile.
    
    Performs comprehensive validation including:
    - Individual holding validation
    - Portfolio diversification checks
    - Risk alignment with user profile
    - Indian market specific validations
    """
    try:
        # Validate portfolio
        validation_response = await validator.validate_portfolio(
            request.portfolio,
            request.user_profile
        )
        
        # Cache validation result
        cache_key = str(hash(str(request.portfolio.holdings) + str(request.user_profile)))
        validation_cache[cache_key] = validation_response
        
        return validation_response
        
    except Exception as e:
        return PortfolioValidationResponse(
            isValid=False,
            errors=[{
                "field": "general",
                "message": f"Validation failed: {str(e)}"
            }],
            warnings=[],
            totalValue=None,
            holdingsCount=None
        )


@router.get("/sample-format")
async def get_sample_format():
    """
    Get sample file format information for user guidance.
    
    Returns format examples, column variations, and usage notes.
    """
    try:
        sample_format = file_parser.get_sample_format()
        return {
            "success": True,
            "data": sample_format
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get sample format: {str(e)}"
        )


@router.post("/bulk-upload")
async def bulk_upload_portfolio(
    request: BulkPortfolioUpload,
    background_tasks: BackgroundTasks
):
    """
    Handle bulk portfolio upload with file parsing and validation.
    
    Processes file in background and returns processing status.
    """
    try:
        # Generate processing ID
        process_id = str(uuid.uuid4())
        
        # Start background processing
        background_tasks.add_task(
            process_bulk_upload,
            process_id,
            request
        )
        
        return {
            "success": True,
            "processId": process_id,
            "message": "File upload started. Check status using process ID."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start bulk upload: {str(e)}"
        )


@router.get("/process-status/{process_id}")
async def get_process_status(process_id: str):
    """
    Get status of background processing task.
    """
    # In a real implementation, you'd check a task queue or database
    # For demo purposes, return a simple response
    return {
        "success": True,
        "processId": process_id,
        "status": "completed",
        "message": "Processing completed successfully"
    }


@router.post("/analyze")
async def analyze_portfolio(
    request: AnalysisRequest
):
    """
    Start comprehensive portfolio analysis.
    
    This endpoint would typically:
    1. Validate the portfolio
    2. Fetch current market data
    3. Generate analysis and recommendations
    4. Return analysis results
    
    For now, returns a mock analysis response.
    """
    try:
        # First validate the portfolio
        validation_response = await validator.validate_portfolio(
            request.portfolio,
            request.user_profile
        )
        
        if not validation_response.isValid:
            return {
                "success": False,
                "message": "Portfolio validation failed",
                "validationErrors": validation_response.errors
            }
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Store request for processing
        portfolios_storage[analysis_id] = request.portfolio
        user_profiles_storage[analysis_id] = request.user_profile
        
        # Return mock analysis (in real implementation, this would trigger actual analysis)
        return {
            "success": True,
            "analysisId": analysis_id,
            "message": "Portfolio analysis started",
            "data": {
                "portfolio": request.portfolio,
                "validation": validation_response,
                "estimatedProcessingTime": "2-3 minutes"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze portfolio: {str(e)}"
        )


@router.get("/suggestions")
async def get_portfolio_suggestions(
    portfolio_data: str = None
):
    """
    Get portfolio improvement suggestions.
    
    Returns actionable suggestions for portfolio optimization.
    """
    try:
        if not portfolio_data:
            return {
                "success": True,
                "suggestions": [
                    "Add at least 3-5 different stocks for better diversification",
                    "Consider stocks from different sectors (Banking, IT, FMCG, etc.)",
                    "Keep individual stock allocation below 20% of total portfolio",
                    "Include both large-cap and mid-cap stocks for balanced growth",
                    "Regular portfolio review and rebalancing is recommended"
                ]
            }
        
        # In real implementation, parse portfolio_data and provide specific suggestions
        return {
            "success": True,
            "suggestions": [
                "Your portfolio shows good diversification",
                "Consider adding exposure to emerging sectors",
                "Review allocation percentages for optimal balance"
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get suggestions: {str(e)}"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def process_bulk_upload(
    process_id: str,
    request: BulkPortfolioUpload
):
    """
    Background task for processing bulk portfolio upload.
    
    In a real implementation, this would:
    1. Process the uploaded file
    2. Validate the data
    3. Store results in database
    4. Send notifications to user
    """
    try:
        # Simulate processing time
        await asyncio.sleep(2)
        
        # Store the user profile
        user_profiles_storage[process_id] = request.user_profile
        
        # In real implementation:
        # - Parse uploaded file
        # - Validate portfolio data
        # - Store in database
        # - Generate analysis
        # - Send completion notification
        
        print(f"Bulk upload processing completed for {process_id}")
        
    except Exception as e:
        print(f"Bulk upload processing failed for {process_id}: {str(e)}")


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "file_parser": "active",
            "validator": "active",
            "storage": "active"
        }
    }


@router.get("/stats")
async def get_stats():
    """
    Get system statistics for monitoring.
    """
    return {
        "success": True,
        "data": {
            "portfolios_stored": len(portfolios_storage),
            "user_profiles_stored": len(user_profiles_storage),
            "validation_cache_size": len(validation_cache),
            "uptime": "N/A",  # In real implementation, track actual uptime
            "last_updated": datetime.now().isoformat()
        }
    }


@router.delete("/clear-cache")
async def clear_cache():
    """
    Clear all cached data (for development/testing).
    """
    try:
        portfolios_storage.clear()
        user_profiles_storage.clear()
        validation_cache.clear()
        
        return {
            "success": True,
            "message": "All cache cleared successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )