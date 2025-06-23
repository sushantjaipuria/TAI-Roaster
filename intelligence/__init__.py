"""
TAI Intelligence Module
Advanced ML-powered portfolio analysis system
"""

__version__ = "1.0.0"

# Export main pipeline functions for easy import
try:
    from .pipeline import run_full_pipeline, run_basic_ml_pipeline
    from .config import get_config
    from .llm_trading_expert import LLMTradingExpert
    
    __all__ = [
        'run_full_pipeline',
        'run_basic_ml_pipeline', 
        'get_config',
        'LLMTradingExpert'
    ]
except ImportError as e:
    # Handle missing dependencies gracefully during initial setup
    print(f"Warning: Some intelligence module dependencies not available: {e}")
    __all__ = [] 