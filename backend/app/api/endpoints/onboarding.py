from fastapi import APIRouter, HTTPException
from typing import Dict
import uuid
from datetime import datetime

from app.models.onboarding import (
    UserProfileRequest, 
    UserProfileResponse, 
    OnboardingSession
)

router = APIRouter()

# In-memory storage for demo purposes
# In production, this would be a proper database
sessions_storage: Dict[str, OnboardingSession] = {}


@router.post("/", response_model=UserProfileResponse)
async def submit_onboarding(user_profile: UserProfileRequest):
    """
    Submit user onboarding information and create a session.
    
    This endpoint collects the user's risk profile, investment preferences,
    and goals to personalize the portfolio analysis.
    """
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session
        session = OnboardingSession(
            session_id=session_id,
            user_profile=user_profile,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # Store session
        sessions_storage[session_id] = session
        
        return UserProfileResponse(
            success=True,
            session_id=session_id,
            message="User profile saved successfully",
            data=user_profile
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save user profile: {str(e)}")


@router.get("/{session_id}", response_model=UserProfileResponse)
async def get_onboarding(session_id: str):
    """
    Retrieve user onboarding information by session ID.
    """
    try:
        if session_id not in sessions_storage:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions_storage[session_id]
        
        return UserProfileResponse(
            success=True,
            session_id=session_id,
            message="User profile retrieved successfully",
            data=session.user_profile
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user profile: {str(e)}")


@router.put("/{session_id}", response_model=UserProfileResponse)
async def update_onboarding(session_id: str, user_profile: UserProfileRequest):
    """
    Update user onboarding information.
    """
    try:
        if session_id not in sessions_storage:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update session
        session = sessions_storage[session_id]
        session.user_profile = user_profile
        session.updated_at = datetime.now().isoformat()
        
        sessions_storage[session_id] = session
        
        return UserProfileResponse(
            success=True,
            session_id=session_id,
            message="User profile updated successfully",
            data=user_profile
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user profile: {str(e)}") 