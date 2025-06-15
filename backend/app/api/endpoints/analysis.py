from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict
import uuid
import asyncio
from datetime import datetime

from app.models.analysis import (
    AnalysisRequest,
    AnalysisRequestResponse,
    AnalysisStatusResponse,
    AnalysisResultResponse,
    AnalysisStatus
)
from app.services.mock_analysis import MockAnalysisService
from app.api.endpoints.onboarding import sessions_storage
from app.api.endpoints.portfolio import portfolios_storage

router = APIRouter()

# In-memory storage for analysis requests
analysis_requests: Dict[str, AnalysisRequest] = {}
analysis_results: Dict[str, any] = {}


@router.post("/predict", response_model=AnalysisRequestResponse)
async def request_analysis(
    background_tasks: BackgroundTasks,
    session_id: str
):
    """
    Request portfolio analysis.
    
    This triggers the analysis process for the user's portfolio based on their
    risk profile and investment preferences.
    """
    try:
        # Validate session exists
        if session_id not in sessions_storage:
            raise HTTPException(status_code=404, detail="User session not found")
        
        # Validate portfolio exists
        if session_id not in portfolios_storage:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            request_id=request_id,
            session_id=session_id,
            status=AnalysisStatus.PENDING,
            created_at=datetime.now(),
            progress=0
        )
        
        # Store request
        analysis_requests[request_id] = analysis_request
        
        # Start analysis in background
        background_tasks.add_task(
            perform_analysis,
            request_id,
            session_id
        )
        
        return AnalysisRequestResponse(
            success=True,
            message="Analysis request submitted successfully",
            request_id=request_id,
            estimated_time=30  # 30 seconds estimated time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to request analysis: {str(e)}")


@router.get("/analysis/{request_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(request_id: str):
    """
    Get the status of an analysis request.
    """
    try:
        if request_id not in analysis_requests:
            raise HTTPException(status_code=404, detail="Analysis request not found")
        
        request = analysis_requests[request_id]
        
        # Calculate estimated remaining time
        estimated_remaining = None
        if request.status == AnalysisStatus.PROCESSING:
            remaining_progress = 100 - request.progress
            estimated_remaining = int(remaining_progress * 0.3)  # Rough estimate
        
        return AnalysisStatusResponse(
            success=True,
            request_id=request_id,
            status=request.status,
            progress=request.progress,
            estimated_remaining=estimated_remaining,
            error_message=request.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis status: {str(e)}")


@router.get("/analysis/{request_id}", response_model=AnalysisResultResponse)
async def get_analysis_results(request_id: str):
    """
    Get the results of a completed analysis.
    """
    try:
        if request_id not in analysis_requests:
            raise HTTPException(status_code=404, detail="Analysis request not found")
        
        request = analysis_requests[request_id]
        
        if request.status != AnalysisStatus.COMPLETED:
            if request.status == AnalysisStatus.FAILED:
                return AnalysisResultResponse(
                    success=False,
                    request_id=request_id,
                    error_message=request.error_message
                )
            else:
                raise HTTPException(
                    status_code=409, 
                    detail=f"Analysis not completed yet. Current status: {request.status}"
                )
        
        # Get analysis results
        if request_id not in analysis_results:
            raise HTTPException(status_code=404, detail="Analysis results not found")
        
        analysis = analysis_results[request_id]
        
        return AnalysisResultResponse(
            success=True,
            request_id=request_id,
            analysis=analysis,
            completed_at=request.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis results: {str(e)}")


@router.delete("/analysis/{request_id}")
async def delete_analysis(request_id: str):
    """
    Delete an analysis request and its results.
    """
    try:
        if request_id not in analysis_requests:
            raise HTTPException(status_code=404, detail="Analysis request not found")
        
        # Delete request and results
        del analysis_requests[request_id]
        if request_id in analysis_results:
            del analysis_results[request_id]
        
        return {
            "success": True,
            "message": "Analysis deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")


@router.get("/analysis/{request_id}/retry")
async def retry_analysis(request_id: str, background_tasks: BackgroundTasks):
    """
    Retry a failed analysis.
    """
    try:
        if request_id not in analysis_requests:
            raise HTTPException(status_code=404, detail="Analysis request not found")
        
        request = analysis_requests[request_id]
        
        if request.status not in [AnalysisStatus.FAILED]:
            raise HTTPException(status_code=409, detail="Analysis can only be retried if it failed")
        
        # Reset request status
        request.status = AnalysisStatus.PENDING
        request.progress = 0
        request.error_message = None
        analysis_requests[request_id] = request
        
        # Start analysis in background
        background_tasks.add_task(
            perform_analysis,
            request_id,
            request.session_id
        )
        
        return {
            "success": True,
            "message": "Analysis retry initiated",
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retry analysis: {str(e)}")


async def perform_analysis(request_id: str, session_id: str):
    """
    Background task to perform portfolio analysis.
    """
    try:
        # Update status to processing
        request = analysis_requests[request_id]
        request.status = AnalysisStatus.PROCESSING
        request.progress = 10
        analysis_requests[request_id] = request
        
        # Get user profile and portfolio
        user_session = sessions_storage[session_id]
        portfolio = portfolios_storage[session_id]
        
        # Initialize mock analysis service
        analysis_service = MockAnalysisService()
        
        # Simulate analysis process with progress updates
        for progress in [25, 50, 75, 90]:
            await asyncio.sleep(2)  # Simulate processing time
            request.progress = progress
            analysis_requests[request_id] = request
        
        # Perform analysis
        analysis_result = await analysis_service.analyze_portfolio(
            portfolio=portfolio,
            user_profile=user_session.user_profile
        )
        
        # Store results
        analysis_results[request_id] = analysis_result
        
        # Update request status
        request.status = AnalysisStatus.COMPLETED
        request.progress = 100
        request.completed_at = datetime.now()
        analysis_requests[request_id] = request
        
    except Exception as e:
        # Handle analysis failure
        request = analysis_requests.get(request_id)
        if request:
            request.status = AnalysisStatus.FAILED
            request.error_message = str(e)
            analysis_requests[request_id] = request 