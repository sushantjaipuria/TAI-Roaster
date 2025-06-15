from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Dict
import pandas as pd
import io
from datetime import datetime

from app.models.portfolio import (
    Portfolio, 
    PortfolioItem, 
    PortfolioUploadResponse,
    PortfolioUpdateRequest,
    PortfolioUpdateResponse
)
from app.services.file_parser import FileParserService
from app.core.config import settings

router = APIRouter()

# In-memory storage for demo purposes
portfolios_storage: Dict[str, Portfolio] = {}


@router.post("/upload", response_model=PortfolioUploadResponse)
async def upload_portfolio(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Upload and parse portfolio CSV/Excel file.
    
    Expected file format:
    - CSV or Excel file
    - Columns: ticker, quantity, avg_price (or similar variations)
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        file_extension = "." + file.filename.split(".")[-1].lower()
        if file_extension not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed types: {settings.ALLOWED_FILE_TYPES}"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Parse file
        parser_service = FileParserService()
        portfolio = await parser_service.parse_file(content, file.filename)
        
        # Store portfolio
        portfolios_storage[session_id] = portfolio
        
        return PortfolioUploadResponse(
            success=True,
            message="Portfolio uploaded and parsed successfully",
            portfolio=portfolio
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return PortfolioUploadResponse(
            success=False,
            message="Failed to process portfolio file",
            errors=[str(e)]
        )


@router.get("/preview/{session_id}")
async def get_portfolio_preview(session_id: str):
    """
    Get portfolio preview for editing.
    """
    try:
        if session_id not in portfolios_storage:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        portfolio = portfolios_storage[session_id]
        
        return {
            "success": True,
            "message": "Portfolio retrieved successfully",
            "data": portfolio
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve portfolio: {str(e)}")


@router.put("/preview/{session_id}", response_model=PortfolioUpdateResponse)
async def update_portfolio_preview(session_id: str, request: PortfolioUpdateRequest):
    """
    Update portfolio data after user edits.
    """
    try:
        if session_id not in portfolios_storage:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Validate and update portfolio
        portfolio = request.portfolio
        portfolio.calculate_total_value()
        portfolio.calculate_allocations()
        portfolio.last_updated = datetime.now()
        
        # Store updated portfolio
        portfolios_storage[session_id] = portfolio
        
        return PortfolioUpdateResponse(
            success=True,
            message="Portfolio updated successfully",
            portfolio=portfolio
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update portfolio: {str(e)}")


@router.delete("/{session_id}")
async def delete_portfolio(session_id: str):
    """
    Delete portfolio data.
    """
    try:
        if session_id not in portfolios_storage:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        del portfolios_storage[session_id]
        
        return {
            "success": True,
            "message": "Portfolio deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete portfolio: {str(e)}")


@router.get("/validate/{session_id}")
async def validate_portfolio(session_id: str):
    """
    Validate portfolio data and return validation results.
    """
    try:
        if session_id not in portfolios_storage:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        portfolio = portfolios_storage[session_id]
        errors = []
        warnings = []
        
        # Basic validation
        if not portfolio.items:
            errors.append("Portfolio is empty")
        
        # Validate individual items
        for item in portfolio.items:
            if not item.ticker:
                errors.append(f"Missing ticker symbol")
            if item.quantity <= 0:
                errors.append(f"Invalid quantity for {item.ticker}")
            if item.avg_price <= 0:
                errors.append(f"Invalid average price for {item.ticker}")
        
        # Check for concentration risk
        if portfolio.total_value > 0:
            for item in portfolio.items:
                if item.allocation and item.allocation > 20:
                    warnings.append(f"{item.ticker} represents {item.allocation:.1f}% of portfolio (high concentration)")
        
        return {
            "success": True,
            "message": "Portfolio validation completed",
            "data": {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "summary": {
                    "total_items": len(portfolio.items),
                    "total_value": portfolio.total_value,
                    "unique_tickers": len(set(item.ticker for item in portfolio.items))
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate portfolio: {str(e)}") 