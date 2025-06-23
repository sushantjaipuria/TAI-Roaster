"""
ML Models Manager for TAI-Roaster
Manages loading and initialization of trained ML models from the intelligence module
"""

import sys
import os
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to Python path for intelligence module imports
project_root = Path(__file__).parent.parent.parent.parent  # Go up to TAI-Roaster root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from intelligence.predictors.base_xgboost import XGBoostPredictor
    from intelligence.predictors.ngboost_predictor import NGBoostPredictor
    from intelligence.predictors.quantile_regressor import QuantileRegressor
    from intelligence.predictors.classifier import ClassifierPredictor
except ImportError as e:
    print(f"Warning: Intelligence module predictors import failed: {e}")
    XGBoostPredictor = None
    NGBoostPredictor = None
    QuantileRegressor = None
    ClassifierPredictor = None

logger = logging.getLogger(__name__)

class ModelManager:
    """Singleton for managing ML models"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.models = {}
            self.model_paths = {}
            self.predictors = {}
            self.initialized = False
            self._setup_model_paths()
            self.load_models()
            ModelManager._initialized = True
    
    def _setup_model_paths(self):
        """Setup paths to all trained models"""
        models_base_path = project_root / "intelligence" / "models"
        enhanced_path = models_base_path / "enhanced"
        
        self.model_paths = {
            # Enhanced models (13 total)
            'xgboost': enhanced_path / 'xgboost_model.pkl',
            'lightgbm': enhanced_path / 'lightgbm_model.pkl',
            'catboost': enhanced_path / 'catboost_model.pkl',
            'ngboost': enhanced_path / 'ngboost_model.pkl',
            'elastic_net': enhanced_path / 'elastic_net_model.pkl',
            'gradient_boosting': enhanced_path / 'gradient_boosting_model.pkl',
            'lasso': enhanced_path / 'lasso_model.pkl',
            'mlp': enhanced_path / 'mlp_model.pkl',
            'random_forest': enhanced_path / 'random_forest_model.pkl',
            'ridge': enhanced_path / 'ridge_model.pkl',
            'svr': enhanced_path / 'svr_model.pkl',
            'preprocessors': enhanced_path / 'preprocessors.pkl',
            'performance_metrics': enhanced_path / 'performance_metrics.pkl',
            
            # Additional models from main models directory
            'quantile_model': models_base_path / 'quantile_model.pkl',
            'classifier': models_base_path / 'classifier.pkl'
        }
    
    def load_models(self):
        """Load all trained models on startup"""
        try:
            logger.info("🤖 Starting ML models loading...")
            
            # Check if model files exist
            missing_models = []
            for model_name, model_path in self.model_paths.items():
                if not model_path.exists():
                    missing_models.append(f"{model_name}: {model_path}")
                    logger.warning(f"⚠️  Model file missing: {model_path}")
            
            if missing_models:
                logger.error(f"❌ Missing {len(missing_models)} model files:")
                for missing in missing_models[:5]:  # Show first 5
                    logger.error(f"   - {missing}")
                if len(missing_models) > 5:
                    logger.error(f"   ... and {len(missing_models) - 5} more")
            
            # Load available models
            loaded_count = 0
            
            # Load enhanced models using joblib
            for model_name in ['xgboost', 'lightgbm', 'catboost', 'ngboost', 
                              'elastic_net', 'gradient_boosting', 'lasso', 'mlp',
                              'random_forest', 'ridge', 'svr', 'preprocessors', 
                              'performance_metrics']:
                try:
                    if self.model_paths[model_name].exists():
                        self.models[model_name] = joblib.load(self.model_paths[model_name])
                        loaded_count += 1
                        logger.debug(f"✅ Loaded {model_name} model")
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name}: {e}")
            
            # Initialize predictor classes if available
            if XGBoostPredictor and 'xgboost' in self.models:
                self.predictors['xgboost'] = XGBoostPredictor()
                logger.debug("✅ XGBoost predictor initialized")
            
            if NGBoostPredictor and 'ngboost' in self.models:
                self.predictors['ngboost'] = NGBoostPredictor()
                logger.debug("✅ NGBoost predictor initialized")
            
            if QuantileRegressor and self.model_paths['quantile_model'].exists():
                self.predictors['quantile'] = QuantileRegressor()
                logger.debug("✅ Quantile regressor initialized")
            
            if ClassifierPredictor and self.model_paths['classifier'].exists():
                self.predictors['classifier'] = ClassifierPredictor()
                logger.debug("✅ Classifier predictor initialized")
            
            self.initialized = True
            logger.info(f"✅ ML models loaded successfully: {loaded_count} models, {len(self.predictors)} predictors")
            
        except Exception as e:
            logger.error(f"❌ Critical error loading ML models: {e}")
            self.initialized = False
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a specific model by name"""
        return self.models.get(model_name)
    
    def get_predictor(self, predictor_name: str) -> Optional[Any]:
        """Get a specific predictor by name"""
        return self.predictors.get(predictor_name)
    
    def get_ensemble_models(self) -> Dict[str, Any]:
        """Get the main ensemble models for predictions"""
        ensemble_models = {}
        for model_name in ['xgboost', 'lightgbm', 'catboost', 'ngboost']:
            if model_name in self.models:
                ensemble_models[model_name] = self.models[model_name]
        return ensemble_models
    
    def get_preprocessors(self) -> Optional[Any]:
        """Get data preprocessors"""
        return self.models.get('preprocessors')
    
    def get_performance_metrics(self) -> Optional[Any]:
        """Get performance metrics data"""
        return self.models.get('performance_metrics')
    
    def is_ready(self) -> bool:
        """Check if models are ready for predictions"""
        return self.initialized and len(self.models) > 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'initialized': self.initialized,
            'total_models': len(self.models),
            'total_predictors': len(self.predictors),
            'loaded_models': list(self.models.keys()),
            'available_predictors': list(self.predictors.keys()),
            'ensemble_ready': len(self.get_ensemble_models()) >= 2
        }
    
    def reload_models(self):
        """Reload all models (useful for updates)"""
        logger.info("🔄 Reloading ML models...")
        self.models.clear()
        self.predictors.clear()
        self.load_models()

# Create singleton instance
model_manager = ModelManager() 