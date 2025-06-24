# intelligence/training/enhanced_model_trainer.py

import sys
import os
import time
import logging
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from ngboost import NGBRegressor

# Hyperparameter optimization
import optuna
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from bayes_opt import BayesianOptimization

# Feature engineering
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer

# Model interpretation
import shap

from intelligence.training.feature_builder import build_features, get_feature_columns
from intelligence.training.data_loader import download_nse_data

# Try to import data config for dynamic settings
try:
    from backend.app.api.data_config import get_current_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("[WARNING] Data config not available, using defaults")

# Setup logging
def setup_logging():
    """Setup comprehensive logging for training process"""
    log_dir = "intelligence/training/logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")

class EnhancedModelTrainer:
    """Enhanced model trainer with comprehensive ML pipeline and detailed logging"""
    
    def __init__(self, target_column: str = "Target_5D_Return"):
        self.target_column = target_column
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_columns = []
        self.model_performances = {}
        self.start_time = time.time()
        
        logger.info("="*60)
        logger.info("ENHANCED MODEL TRAINER INITIALIZED")
        logger.info("="*60)
        logger.info(f"Target column: {target_column}")
        logger.info(f"Start time: {datetime.now()}")
        
    def log_progress(self, stage: str, message: str = ""):
        """Log progress with timing information"""
        elapsed = time.time() - self.start_time
        logger.info(f"[{stage}] {message} (Elapsed: {elapsed:.1f}s)")
        
    def prepare_training_data(self, tickers: List[str], config=None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare comprehensive training dataset with configurable dates"""
        self.log_progress("DATA_PREP", f"Preparing training data for {len(tickers)} tickers...")
        
        # Get data configuration
        from datetime import datetime, timedelta
        if config is None:
            try:
                if CONFIG_AVAILABLE:
                    data_config = get_current_config()
                    start_date = data_config.training.start_date.strftime("%Y-%m-%d")
                    end_date = data_config.training.end_date.strftime("%Y-%m-%d") 
                    timeframe = data_config.training.timeframe.value
                else:
                    # Fallback to current hardcoded values
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d")
                    timeframe = "1day"
            except Exception as e:
                # Error fallback
                logger.warning(f"Using fallback configuration: {e}")
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d")
                timeframe = "1day"
        else:
            start_date = config.get('start_date', (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d"))
            end_date = config.get('end_date', datetime.now().strftime("%Y-%m-%d"))
            timeframe = config.get('timeframe', '1day')
        
        logger.info(f"Training period: {start_date} to {end_date}, timeframe: {timeframe}")
        
        all_features = []
        all_targets = []
        
        for i, ticker in enumerate(tickers):
            try:
                self.log_progress("DATA_PREP", f"Processing ticker {i+1}/{len(tickers)}: {ticker}")
                
                # Download data with timeframe
                df_dict = download_nse_data([ticker], start=start_date, end=end_date, timeframe=timeframe)
                prices = df_dict.get(ticker)
        
                if prices is None or len(prices) < 100:
                    logger.warning(f"Insufficient data for {ticker}")
                    continue
                
                # Build features
                features_df = build_features(prices)
                
                if len(features_df) < 50:
                    logger.warning(f"Insufficient features for {ticker}")
                    continue
                
                # Add ticker as feature
                features_df['ticker_encoded'] = hash(ticker) % 1000
                
                # Separate features and target
                target_data = features_df[self.target_column]
                feature_data = features_df.drop(columns=[self.target_column])
                
                all_features.append(feature_data)
                all_targets.append(target_data)
                
                logger.info(f"✓ Processed {ticker}: {len(feature_data)} samples")
                
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data found")
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        self.log_progress("DATA_PREP", f"Combined dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def preprocess_features(self, X: pd.DataFrame, y: pd.Series, fit_preprocessors: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Advanced feature preprocessing with logging"""
        self.log_progress("PREPROCESS", "Starting feature preprocessing...")
        
        if fit_preprocessors:
            # Remove constant and duplicate features
            self.log_progress("PREPROCESS", "Removing constant and duplicate features...")
            self.constant_remover = DropConstantFeatures()
            self.duplicate_remover = DropDuplicateFeatures()
            
            X = self.constant_remover.fit_transform(X)
            X = self.duplicate_remover.fit_transform(X)
            
            # Handle missing values
            self.log_progress("PREPROCESS", "Handling missing values...")
            self.imputer = MeanMedianImputer(imputation_method='median')
            X = self.imputer.fit_transform(X)
            
            # Handle outliers - exclude binary pattern features
            pattern_cols = [col for col in X.columns if col.startswith('CDL_')]
            non_pattern_cols = [col for col in X.columns if not col.startswith('CDL_')]
            
            if non_pattern_cols:
                self.log_progress("PREPROCESS", "Handling outliers...")
                self.outlier_handler = Winsorizer(
                    variables=non_pattern_cols,
                    capping_method='iqr', 
                    tail='both', 
                    fold=1.5
                )
                X = self.outlier_handler.fit_transform(X)
        
            # Scale features - also exclude pattern features from scaling
            self.log_progress("PREPROCESS", "Scaling features...")
            self.scaler = RobustScaler()
            if non_pattern_cols:
                X_scaled = X.copy()
                X_scaled[non_pattern_cols] = self.scaler.fit_transform(X[non_pattern_cols])
            else:
                X_scaled = X.copy()
            
        else:
            # Transform using fitted preprocessors
            X = self.constant_remover.transform(X)
            X = self.duplicate_remover.transform(X)
            X = self.imputer.transform(X)
            
            # Apply outlier handling only to non-pattern features
            pattern_cols = [col for col in X.columns if col.startswith('CDL_')]
            non_pattern_cols = [col for col in X.columns if not col.startswith('CDL_')]
            
            if non_pattern_cols and hasattr(self, 'outlier_handler'):
                X = self.outlier_handler.transform(X)
            
            # Apply scaling only to non-pattern features
            if non_pattern_cols and hasattr(self, 'scaler'):
                X_scaled = X.copy()
                X_scaled[non_pattern_cols] = self.scaler.transform(X[non_pattern_cols])
            else:
                X_scaled = X.copy()
        
        # Remove extreme target outliers
        q_low = y.quantile(0.01)
        q_high = y.quantile(0.99)
        mask = (y >= q_low) & (y <= q_high)
        
        X_processed = X_scaled[mask]
        y_processed = y[mask]
        
        self.log_progress("PREPROCESS", f"Final processed data: {X_processed.shape[0]} samples")
        
        return X_processed, y_processed
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = "rfe", n_features: int = 50) -> pd.DataFrame:
        """Feature selection with logging and fallback for speed"""
        self.log_progress("FEATURE_SELECT", f"Selecting {n_features} features using {method}...")
        import time
        import signal
        start_time = time.time()
        fallback_to_kbest = False
        X_selected = None
        selected_features = None
        
        # Force SelectKBest for large datasets to avoid hanging
        if X.shape[0] > 1000 or X.shape[1] > 50:
            self.log_progress("FEATURE_SELECT", f"Dataset is large ({X.shape[0]} samples, {X.shape[1]} features), using SelectKBest for speed and stability.")
            method = "kbest"
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Feature selection timed out")
        
        # Set timeout for feature selection (2 minutes)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)
        
        try:
            if method == "rfe":
                try:
                    self.log_progress("FEATURE_SELECT", "Attempting RFE with RandomForest...")
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    selector = RFE(estimator, n_features_to_select=n_features, step=0.2)
                    X_selected = selector.fit_transform(X, y)
                    selected_features = X.columns[selector.support_]
                    self.log_progress("FEATURE_SELECT", "RFE completed successfully")
                except Exception as e:
                    self.log_progress("FEATURE_SELECT", f"RFE failed: {e}, falling back to SelectKBest.")
                    fallback_to_kbest = True
            
            if method == "kbest" or fallback_to_kbest:
                try:
                    self.log_progress("FEATURE_SELECT", "Using SelectKBest with f_regression...")
                    selector = SelectKBest(score_func=f_regression, k=min(n_features, X.shape[1]))
                    X_selected = selector.fit_transform(X, y)
                    selected_features = X.columns[selector.get_support()]
                    self.log_progress("FEATURE_SELECT", "SelectKBest completed successfully")
                except Exception as e:
                    self.log_progress("FEATURE_SELECT", f"SelectKBest failed: {e}, using all features.")
                    X_selected = X
                    selected_features = X.columns
            
            if X_selected is None or selected_features is None:
                self.log_progress("FEATURE_SELECT", "No feature selection performed, using all features.")
                X_selected = X
                selected_features = X.columns
                
        except TimeoutError:
            self.log_progress("FEATURE_SELECT", "Feature selection timed out, using SelectKBest...")
            try:
                selector = SelectKBest(score_func=f_regression, k=min(n_features, X.shape[1]))
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()]
                self.log_progress("FEATURE_SELECT", "SelectKBest completed after timeout")
            except Exception as e:
                self.log_progress("FEATURE_SELECT", f"SelectKBest also failed: {e}, using all features.")
                X_selected = X
                selected_features = X.columns
        finally:
            signal.alarm(0)  # Cancel the alarm
        
        self.feature_columns = selected_features.tolist()
        X_selected_df = pd.DataFrame(X_selected, columns=self.feature_columns, index=X.index)
        elapsed = time.time() - start_time
        self.log_progress("FEATURE_SELECT", f"Selected {len(self.feature_columns)} features in {elapsed:.2f}s")
        return X_selected_df
    
    def train_xgboost_optimized(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train XGBoost with reduced optimization for faster training"""
        self.log_progress("XGBOOST", "Starting XGBoost training with optimization...")
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),  # Reduced range
                'max_depth': trial.suggest_int('max_depth', 3, 8),  # Reduced range
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),  # Reduced range
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),  # Reduced range
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),  # Reduced range
                'random_state': 42
            }
            
            # Simplified validation - use single train/test split instead of CV
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)
            
            return score
        
        # Reduced number of trials and timeout
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(objective, n_trials=10, timeout=300)  # 10 trials, 5 min timeout
        except Exception as e:
            logger.warning(f"XGBoost optimization failed: {e}, using default parameters")
            # Use default parameters if optimization fails
            best_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42
            }
        else:
            best_params = study.best_params
            best_params['random_state'] = 42
        
        # Train final model
        self.log_progress("XGBOOST", "Training final XGBoost model...")
        model = xgb.XGBRegressor(**best_params)
        model.fit(X, y)
        
        self.models['xgboost'] = model
        
        self.log_progress("XGBOOST", "XGBoost training completed")
        
        return {
            'model': model,
            'best_params': best_params,
            'best_score': study.best_value if 'study' in locals() else None
        }
    
    def train_lightgbm_optimized(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train LightGBM with reduced optimization"""
        self.log_progress("LIGHTGBM", "Starting LightGBM training with optimization...")
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),  # Reduced range
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),  # Reduced range
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),  # Reduced range
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  # Reduced range
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),  # Reduced range
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),  # Reduced range
                'random_state': 42,
                'verbose': -1
            }
            
            # Simplified validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
            
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)
            
            return score
        
        # Reduced optimization
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(objective, n_trials=10, timeout=300)  # 10 trials, 5 min timeout
        except Exception as e:
            logger.warning(f"LightGBM optimization failed: {e}, using default parameters")
            best_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'min_child_samples': 20,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42,
                'verbose': -1
            }
        else:
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['verbose'] = -1
        
        # Train final model
        self.log_progress("LIGHTGBM", "Training final LightGBM model...")
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X, y)
        
        self.models['lightgbm'] = model
        
        self.log_progress("LIGHTGBM", "LightGBM training completed")
        
        return {
            'model': model,
            'best_params': best_params,
            'best_score': study.best_value if 'study' in locals() else None
        }
    
    def train_catboost(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train CatBoost model with logging"""
        self.log_progress("CATBOOST", "Starting CatBoost training...")
        
        model = CatBoostRegressor(
            iterations=500,  # Reduced from 1000
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        
        model.fit(X, y)
        self.models['catboost'] = model
        
        self.log_progress("CATBOOST", "CatBoost training completed")
        
        return {'model': model}
    
    def train_ngboost(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train NGBoost with simplified approach"""
        self.log_progress("NGBOOST", "Starting NGBoost training...")
        
        try:
            # Clean data
            X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            y_clean = y.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Ensure y is 1D
            if len(y_clean.shape) > 1:
                y_clean = y_clean.squeeze()
            
            self.log_progress("NGBOOST", f"Data shapes - X: {X_clean.shape}, y: {y_clean.shape}")
            
            model = NGBRegressor(
                n_estimators=200,  # Reduced from 500
                learning_rate=0.01,
                random_state=42,
                verbose=False
            )
            
            model.fit(X_clean.values, y_clean.values)
            self.models['ngboost'] = model
            
            self.log_progress("NGBOOST", "NGBoost training completed")
            
            return {'model': model}
            
        except Exception as e:
            logger.error(f"NGBoost training failed: {e}")
            return {'model': None, 'error': str(e)}
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train ensemble of traditional ML models with logging"""
        self.log_progress("ENSEMBLE", "Starting ensemble model training...")
        
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),  # Reduced from 200
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),  # Reduced from 200
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'svr': SVR(kernel='rbf', C=1.0),
            'mlp': MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=300)  # Reduced complexity
        }
        
        ensemble_results = {}
        
        for name, model in models.items():
            try:
                self.log_progress("ENSEMBLE", f"Training {name}...")
                model.fit(X, y)
                self.models[name] = model
                
                # Simplified evaluation
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')  # Reduced CV folds
                ensemble_results[name] = {
                    'model': model,
                    'cv_score': -cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"✓ {name}: CV RMSE = {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
        
        self.log_progress("ENSEMBLE", "Ensemble training completed")
        
        return ensemble_results
    
    def evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Comprehensive model evaluation with logging"""
        self.log_progress("EVALUATION", "Starting model evaluation...")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            try:
                self.log_progress("EVALUATION", f"Evaluating {name}...")
                
                if name == 'ngboost':
                    y_pred = model.predict(X_test.values)
                else:
                    y_pred = model.predict(X_test)
                
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                }
                
                evaluation_results[name] = metrics
                logger.info(f"✓ {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
        
        self.model_performances = evaluation_results
        self.log_progress("EVALUATION", "Model evaluation completed")
        
        return evaluation_results
    
    def generate_feature_importance(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate feature importance for interpretability"""
        self.log_progress("FEATURE_IMPORTANCE", "Generating feature importance...")
        
        importance_results = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    importance_results[name] = importance
                    
                elif name in ['xgboost', 'lightgbm', 'catboost']:
                    # Use SHAP for tree-based models
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X.sample(min(500, len(X))))  # Reduced sample size
                    
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': np.abs(shap_values).mean(0)
                    }).sort_values('importance', ascending=False)
                    
                    importance_results[name] = importance
            
            except Exception as e:
                logger.error(f"Failed to generate importance for {name}: {e}")
        
        self.log_progress("FEATURE_IMPORTANCE", "Feature importance generation completed")
        
        return importance_results
    
    def save_models(self, save_dir: str = "intelligence/models/enhanced/"):
        """Save all trained models and preprocessors"""
        os.makedirs(save_dir, exist_ok=True)
        
        self.log_progress("SAVE", f"Saving models to {save_dir}...")
        
        # Save models
        for name, model in self.models.items():
            if model is not None:
                model_path = f"{save_dir}{name}_model.pkl"
                joblib.dump(model, model_path)
                logger.info(f"✓ Saved {name} to {model_path}")
        
        # Save preprocessors
        preprocessors = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'outlier_handler': self.outlier_handler,
            'constant_remover': self.constant_remover,
            'duplicate_remover': self.duplicate_remover,
            'feature_selectors': self.feature_selectors,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(preprocessors, f"{save_dir}preprocessors.pkl")
        
        # Save performance metrics
        joblib.dump(self.model_performances, f"{save_dir}performance_metrics.pkl")
        
        self.log_progress("SAVE", "All models and preprocessors saved successfully!")
    
    def train_all_models(self, tickers: List[str], config=None) -> Dict[str, Any]:
        """Complete training pipeline with comprehensive logging"""
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Prepare data
            X, y = self.prepare_training_data(tickers, config)
            
            # Preprocess features
            X_processed, y_processed = self.preprocess_features(X, y, fit_preprocessors=True)
            
            # Feature selection - Force SelectKBest for large datasets to avoid hanging
            if X_processed.shape[0] > 1000 or X_processed.shape[1] > 50:
                logger.info(f"Large dataset detected ({X_processed.shape[0]} samples, {X_processed.shape[1]} features), using SelectKBest for speed and stability")
                X_selected = self.select_features(X_processed, y_processed, method="kbest", n_features=50)
            else:
                X_selected = self.select_features(X_processed, y_processed, method="rfe", n_features=50)
            
            # Train models
            results = {}
            
            # Tree-based models with optimization
            results['xgboost'] = self.train_xgboost_optimized(X_selected, y_processed)
            results['lightgbm'] = self.train_lightgbm_optimized(X_selected, y_processed)
            results['catboost'] = self.train_catboost(X_selected, y_processed)
            results['ngboost'] = self.train_ngboost(X_selected, y_processed)
            
            # Ensemble models
            results['ensemble'] = self.train_ensemble_models(X_selected, y_processed)
            
            # Evaluate all models
            evaluation_results = self.evaluate_models(X_selected, y_processed)
            
            # Generate feature importance
            importance_results = self.generate_feature_importance(X_selected)
            
            # Save everything
            self.save_models()
            
            total_time = time.time() - self.start_time
            logger.info("="*60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            logger.info("="*60)
            
            return {
                'training_results': results,
                'evaluation_results': evaluation_results,
                'feature_importance': importance_results,
                'data_shape': X_selected.shape,
                'feature_columns': self.feature_columns,
                'total_time': total_time
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

def main():
    """Main training function with error handling"""
    logger.info("Starting main training function...")
    
    # Sample tickers for training
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS"
    ]
    
    try:
        trainer = EnhancedModelTrainer()
        results = trainer.train_all_models(tickers)
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        
        for model_name, metrics in results['evaluation_results'].items():
            logger.info(f"{model_name.upper()}:")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  R²: {metrics['r2']:.4f}")
            logger.info(f"  MAE: {metrics['mae']:.4f}")
            logger.info("")
            
    except Exception as e:
        logger.error(f"Main training function failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 