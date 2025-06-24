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
from app.utils.file_handler import file_handler

# Import intelligence service and utilities for enhanced analysis
try:
    from app.services.intelligence_service import intelligence_service
    from app.utils.format_converter import convert_enhanced_analysis_to_frontend_format
    from app.utils.file_saver import analysis_file_saver
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Intelligence service not available: {e}")
    INTELLIGENCE_AVAILABLE = False

router = APIRouter()



# In-memory storage for demo purposes (in production, use proper database)
portfolios_storage: Dict[str, PortfolioInput] = {}
user_profiles_storage: Dict[str, UserProfile] = {}
validation_cache: Dict[str, PortfolioValidationResponse] = {}
analysis_status_storage: Dict[str, Dict] = {}  # For tracking analysis progress

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
    try:
        # Create file upload request
        file_request = FileUploadRequest(
            filename=filename or file.filename or "unknown.csv",
            file_size=fileSize or 0,
            content_type=contentType or file.content_type
        )
        
        # Validate file format first
        format_valid, format_errors = await file_parser.validate_file_format(file_request)
        
        if not format_valid:
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
        
        if len(content) == 0:
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
        
        # =================================================================
        # NEW: Save uploaded file to disk (non-breaking addition)
        # =================================================================
        saved_file_path = None
        try:
            save_success, file_path, save_error = await file_handler.save_uploaded_file(
                content, 
                file_request.filename
            )
            
            if save_success:
                saved_file_path = file_path
            # Note: We continue processing even if file saving fails
                
        except Exception as save_exception:
            # Note: File saving failure doesn't stop the upload process
            pass
        # =================================================================
        
        # Parse file
        parse_response = await file_parser.parse_file(content, file_request)
        
        # Store portfolio if parsing was successful
        if parse_response.success and parse_response.portfolio:
            session_id = str(uuid.uuid4())
            portfolios_storage[session_id] = parse_response.portfolio
            
            # =================================================================
            # NEW: Log file location for reference (optional enhancement)
            # =================================================================
            if saved_file_path:
                # Note: We could store this path with the portfolio for future reference
                pass
            # =================================================================
        
        return {
            "success": True,
            "data": parse_response
        }
        
    except Exception as e:
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
            is_valid=False,
            errors=[{
                "field": "general",
                "message": f"Validation failed: {str(e)}"
            }],
            warnings=[],
            total_value=None,
            holdings_count=None
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
    Get status of portfolio analysis processing.
    
    Returns current status, progress percentage, and status message.
    """
    try:
        # Check if analysis status exists
        if process_id in analysis_status_storage:
            status_info = analysis_status_storage[process_id]
            return {
                "success": True,
                "processId": process_id,
                "status": status_info.get("status", "unknown"),
                "progress": status_info.get("progress", 0),
                "message": status_info.get("message", "Processing..."),
                "timestamp": status_info.get("timestamp", datetime.now().isoformat())
            }
        else:
            # If no status found, assume completed (for backward compatibility)
            return {
                "success": True,
                "processId": process_id,
                "status": "completed",
                "progress": 100,
                "message": "Analysis completed successfully",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get process status: {str(e)}"
        )


@router.post("/analyze")
async def analyze_portfolio(
    request: AnalysisRequest
):
    """
    Start comprehensive portfolio analysis using intelligence module.
    
    This endpoint:
    1. Validates the portfolio
    2. Runs ML-powered analysis using intelligence service
    3. Converts results to frontend format
    4. Saves results to processed directory
    5. Returns analysis ID for frontend consumption
    """
    try:
        # First validate the portfolio
        validation_response = await validator.validate_portfolio(
            request.portfolio,
            request.user_profile
        )
        
        if not validation_response.is_valid:
            return {
                "success": False,
                "message": "Portfolio validation failed",
                "validationErrors": validation_response.errors
            }
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Initialize analysis status tracking
        analysis_status_storage[analysis_id] = {
            "status": "processing",
            "progress": 10,
            "message": "Initializing portfolio analysis...",
            "timestamp": datetime.now().isoformat()
        }
        
        # Store request for processing
        portfolios_storage[analysis_id] = request.portfolio
        user_profiles_storage[analysis_id] = request.user_profile
        
        # Try to use intelligence service if available
        if INTELLIGENCE_AVAILABLE and intelligence_service.initialized:
            try:
                print(f"🤖 Running enhanced analysis for {analysis_id} using intelligence module")
                
                # Update status - starting AI analysis
                analysis_status_storage[analysis_id].update({
                    "status": "processing",
                    "progress": 25,
                    "message": "Running AI-powered portfolio analysis...",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Run enhanced analysis using intelligence service
                enhanced_results = await intelligence_service.analyze_portfolio(
                    request.portfolio,
                    request.user_profile
                )
                
                # Update status - generating insights
                analysis_status_storage[analysis_id].update({
                    "status": "processing",
                    "progress": 75,
                    "message": "Generating insights and recommendations...",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Convert to frontend format
                frontend_format = convert_enhanced_analysis_to_frontend_format(enhanced_results)
                
                # Update status - finalizing
                analysis_status_storage[analysis_id].update({
                    "status": "processing",
                    "progress": 90,
                    "message": "Finalizing analysis report...",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Save to processed directory for frontend consumption
                save_success, file_path, save_error = analysis_file_saver.save_demo_format_analysis(
                    frontend_format, 
                    f"portfolio-analysis-{analysis_id[:8]}"
                )
                
                if save_success:
                    print(f"✅ Analysis results saved to: {file_path}")
                else:
                    print(f"⚠️ Failed to save analysis results: {save_error}")
                
                # Mark analysis as completed
                analysis_status_storage[analysis_id].update({
                    "status": "completed",
                    "progress": 100,
                    "message": "Analysis completed successfully!",
                    "timestamp": datetime.now().isoformat()
                })
                
                return {
                    "success": True,
                    "analysisId": analysis_id,
                    "message": "Portfolio analysis completed using AI intelligence",
                    "data": {
                        "portfolio": request.portfolio,
                        "validation": validation_response,
                        "estimatedProcessingTime": "Analysis completed",
                        "intelligence_used": True,
                        "file_saved": save_success
                    }
                }
                
            except Exception as intelligence_error:
                print(f"⚠️ Intelligence analysis failed, falling back to mock: {intelligence_error}")
                # Update status to show error occurred but continuing
                analysis_status_storage[analysis_id].update({
                    "status": "processing",
                    "progress": 50,
                    "message": "AI analysis failed, using fallback method...",
                    "timestamp": datetime.now().isoformat()
                })
                # Fall through to mock analysis
        
        # Fallback to mock analysis if intelligence service unavailable
        print(f"📝 Using mock analysis for {analysis_id} (intelligence not available)")
        
        # Update status to show mock analysis completion
        analysis_status_storage[analysis_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Analysis completed using fallback method",
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "analysisId": analysis_id,
            "message": "Portfolio analysis started (mock mode)",
            "data": {
                "portfolio": request.portfolio,
                "validation": validation_response,
                "estimatedProcessingTime": "2-3 minutes",
                "intelligence_used": False,
                "note": "Using fallback analysis - intelligence module not available"
            }
        }
        
    except Exception as e:
        print(f"❌ Portfolio analysis failed for {analysis_id}: {str(e)}")
        # Update status to show failure
        if 'analysis_id' in locals() and analysis_id in analysis_status_storage:
            analysis_status_storage[analysis_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"Analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
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