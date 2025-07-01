from fastapi import APIRouter

from app.api.endpoints import onboarding, portfolio, analysis, portfolio_performance

api_router = APIRouter()
 
api_router.include_router(onboarding.router, prefix="/onboarding", tags=["onboarding"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(analysis.router, prefix="", tags=["analysis"])
api_router.include_router(portfolio_performance.router, prefix="/portfolio-performance", tags=["portfolio-performance"]) 