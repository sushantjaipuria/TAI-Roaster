from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict
import uuid
import asyncio
import json
import os
from datetime import datetime
import re

from app.models.analysis import (
    AnalysisRequest,
    AnalysisRequestResponse,
    AnalysisStatusResponse,
    AnalysisResultResponse,
    AnalysisStatus
)
from app.api.endpoints.onboarding import sessions_storage
from app.api.endpoints.portfolio import portfolios_storage

router = APIRouter()

# In-memory storage for analysis requests (kept for compatibility)
analysis_requests: Dict[str, AnalysisRequest] = {}
analysis_results: Dict[str, any] = {}


def camel_to_snake(data):
    """
    Convert camelCase keys to snake_case recursively for JSON data.
    Also handles specific format conversions for compatibility.
    """
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            # Convert camelCase to snake_case
            snake_key = re.sub(r'([A-Z])', r'_\1', key).lower()
            
            # Special handling for insights field
            if snake_key == 'insights' and isinstance(value, list):
                # Convert list of insights to a dictionary format
                # Use generic keys for the list items
                insights_dict = {}
                for i, insight in enumerate(value):
                    insights_dict[f"insight_{i}"] = insight
                new_data[snake_key] = insights_dict
            else:
                new_data[snake_key] = camel_to_snake(value)
        return new_data
    elif isinstance(data, list):
        return [camel_to_snake(item) for item in data]
    else:
        return data


def load_demo_analysis():
    """
    Load the demo portfolio analysis data.
    """
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "processed")
    json_file_path = os.path.join(processed_dir, "analysis_demo-portfolio-analysis.json")
    
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid demo analysis JSON file format")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load demo analysis file: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail="Demo analysis file not found")


@router.post("/analysis/request", response_model=AnalysisRequestResponse)
async def request_analysis(
    background_tasks: BackgroundTasks,
    session_id: str,
    analysis_type: str = "comprehensive"
):
    """
    Request a new portfolio analysis.
    Now returns demo analysis immediately instead of processing in background.
    """
    try:
        # Check if session exists
        if session_id not in sessions_storage:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session_id not in portfolios_storage:
            raise HTTPException(status_code=404, detail="Portfolio data not found for session")
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Create analysis request (for compatibility)
        request = AnalysisRequest(
            request_id=request_id,
            session_id=session_id,
            status=AnalysisStatus.COMPLETED,  # Immediately mark as completed
            progress=100,
            completed_at=datetime.now()
        )
        
        # Store request
        analysis_requests[request_id] = request
        
        # Load and store demo analysis results immediately
        demo_analysis = load_demo_analysis()
        analysis_results[request_id] = demo_analysis
        
        return AnalysisRequestResponse(
            success=True,
            request_id=request_id,
            message="Analysis completed using demo data",
            estimated_processing_time="0 seconds"
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
        
        # Since we now return demo data immediately, all analyses are completed
        return AnalysisStatusResponse(
            success=True,
            request_id=request_id,
            status=AnalysisStatus.COMPLETED,
            progress=100,
            estimated_remaining=None,
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
    Always returns demo portfolio analysis data.
    """
    try:
        # Load demo analysis data
        demo_analysis = load_demo_analysis()
        
        # Return demo analysis data for any request
        from fastapi.responses import JSONResponse
        return JSONResponse(content={
            "success": True,
            "request_id": request_id,
            "analysis": demo_analysis,  # Keep original camelCase for frontend
            "completed_at": datetime.now().isoformat()
        })
        
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