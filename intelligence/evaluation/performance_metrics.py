"""
Performance Metrics - Comprehensive Trading & Model Performance Calculator
Calculates all performance metrics specified in the PRD for trading evaluation.

Metrics calculated:
- Return accuracy metrics (RMSE, correlation, accuracy score)
- Directional accuracy (up/down movement prediction)
- Portfolio performance (Sharpe ratio, max drawdown, CAGR, volatility)
- Risk metrics (risk-reward ratio, beta, alpha)
- Statistical measures for comprehensive evaluation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for trading systems.
    
    Features:
    - Return prediction accuracy metrics
    - Directional accuracy analysis
    - Portfolio performance metrics
    - Risk-adjusted performance measures
    - Benchmark comparison metrics
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate (float): Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = risk_free_rate / 252  # Convert to daily
        
        logger.info(f"PerformanceMetrics initialized with risk-free rate: {risk_free_rate:.2%}")
    
    def calculate_return_accuracy(self, predicted_returns: List[float], 
                                 actual_returns: List[float]) -> Dict[str, float]:
        """
        Calculate return prediction accuracy metrics.
        
        Args:
            predicted_returns (List[float]): Model predicted returns
            actual_returns (List[float]): Actual market returns
            
        Returns:
            Dict[str, float]: Return accuracy metrics
        """
        try:
            if len(predicted_returns) == 0 or len(actual_returns) == 0:
                logger.warning("Empty returns data for accuracy calculation")
                return {
                    "rmse": 0.0,
                    "mae": 0.0,
                    "correlation": 0.0,
                    "accuracy_score": 0.0,
                    "r_squared": 0.0
                }
            
            pred_array = np.array(predicted_returns)
            actual_array = np.array(actual_returns)
            
            # Ensure same length
            min_length = min(len(pred_array), len(actual_array))
            pred_array = pred_array[:min_length]
            actual_array = actual_array[:min_length]
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(actual_array, pred_array))
            mae = np.mean(np.abs(actual_array - pred_array))
            
            # Correlation
            if np.std(pred_array) > 0 and np.std(actual_array) > 0:
                correlation = np.corrcoef(pred_array, actual_array)[0, 1]
            else:
                correlation = 0.0
            
            # R-squared
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
            
            # Accuracy score (percentage of predictions within reasonable tolerance)
            tolerance = np.std(actual_array) * 0.5  # 50% of standard deviation
            accurate_predictions = np.abs(pred_array - actual_array) <= tolerance
            accuracy_score_pct = np.mean(accurate_predictions) * 100
            
            metrics = {
                "rmse": float(rmse),
                "mae": float(mae),
                "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                "accuracy_score": float(accuracy_score_pct),
                "r_squared": float(r_squared),
                "tolerance_used": float(tolerance),
                "sample_size": int(min_length)
            }
            
            logger.debug(f"Return accuracy calculated: RMSE={rmse:.4f}, Corr={correlation:.3f}, Acc={accuracy_score_pct:.1f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate return accuracy: {e}")
            return {
                "rmse": 0.0,
                "mae": 0.0,
                "correlation": 0.0,
                "accuracy_score": 0.0,
                "r_squared": 0.0
            }
    
    def calculate_directional_accuracy(self, predicted_returns: List[float],
                                     actual_returns: List[float]) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics.
        
        Args:
            predicted_returns (List[float]): Model predicted returns
            actual_returns (List[float]): Actual market returns
            
        Returns:
            Dict[str, float]: Directional accuracy metrics
        """
        try:
            if len(predicted_returns) == 0 or len(actual_returns) == 0:
                logger.warning("Empty returns data for directional accuracy calculation")
                return {
                    "overall_accuracy": 0.0,
                    "up_movement_accuracy": 0.0,
                    "down_movement_accuracy": 0.0,
                    "neutral_movement_accuracy": 0.0
                }
            
            pred_array = np.array(predicted_returns)
            actual_array = np.array(actual_returns)
            
            # Ensure same length
            min_length = min(len(pred_array), len(actual_array))
            pred_array = pred_array[:min_length]
            actual_array = actual_array[:min_length]
            
            # Define directions (with neutral zone)
            neutral_threshold = 0.001  # 0.1% threshold for neutral
            
            def get_direction(returns):
                directions = np.where(returns > neutral_threshold, 1,  # Up
                                    np.where(returns < -neutral_threshold, -1, 0))  # Down or Neutral
                return directions
            
            pred_directions = get_direction(pred_array)
            actual_directions = get_direction(actual_array)
            
            # Overall accuracy
            overall_accuracy = np.mean(pred_directions == actual_directions) * 100
            
            # Cap directional accuracy at 100%
            overall_accuracy = min(100.0, overall_accuracy)
            
            # Movement-specific accuracies
            up_mask = actual_directions == 1
            down_mask = actual_directions == -1
            neutral_mask = actual_directions == 0
            
            up_accuracy = np.mean(pred_directions[up_mask] == actual_directions[up_mask]) * 100 if np.sum(up_mask) > 0 else 0.0
            down_accuracy = np.mean(pred_directions[down_mask] == actual_directions[down_mask]) * 100 if np.sum(down_mask) > 0 else 0.0
            neutral_accuracy = np.mean(pred_directions[neutral_mask] == actual_directions[neutral_mask]) * 100 if np.sum(neutral_mask) > 0 else 0.0
            
            # Cap all accuracy values at 100%
            up_accuracy = min(100.0, up_accuracy)
            down_accuracy = min(100.0, down_accuracy)
            neutral_accuracy = min(100.0, neutral_accuracy)
            
            metrics = {
                "overall_accuracy": float(overall_accuracy),
                "up_movement_accuracy": float(up_accuracy),
                "down_movement_accuracy": float(down_accuracy),
                "neutral_movement_accuracy": float(neutral_accuracy),
                "up_movements": int(np.sum(up_mask)),
                "down_movements": int(np.sum(down_mask)),
                "neutral_movements": int(np.sum(neutral_mask)),
                "neutral_threshold": float(neutral_threshold),
                "sample_size": int(min_length)
            }
            
            logger.debug(f"Directional accuracy calculated: Overall={overall_accuracy:.1f}%, Up={up_accuracy:.1f}%, Down={down_accuracy:.1f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate directional accuracy: {e}")
            return {
                "overall_accuracy": 0.0,
                "up_movement_accuracy": 0.0,
                "down_movement_accuracy": 0.0,
                "neutral_movement_accuracy": 0.0
            }
    
    def calculate_portfolio_performance(self, portfolio_returns: List[float]) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Args:
            portfolio_returns (List[float]): Daily portfolio returns
            
        Returns:
            Dict[str, Any]: Portfolio performance metrics
        """
        try:
            if len(portfolio_returns) == 0:
                logger.warning("Empty portfolio returns for performance calculation")
                return self._empty_portfolio_metrics()
            
            returns_array = np.array(portfolio_returns)
            
            # Basic statistics
            total_return = (np.prod(1 + returns_array) - 1) * 100
            mean_return = np.mean(returns_array)
            
            # Handle volatility calculation for small samples
            if len(returns_array) <= 1:
                volatility = 0.0
                logger.warning(f"Cannot calculate volatility with {len(returns_array)} data points")
            else:
                volatility = np.std(returns_array, ddof=1) * np.sqrt(252) * 100  # Annualized volatility with sample correction
            
            # CAGR calculation - more robust for small time periods
            trading_days = len(returns_array)
            years = trading_days / 252
            
            # Improved CAGR calculation for edge cases
            if trading_days <= 1:
                # For single day, use total return as approximation
                cagr = total_return
                logger.warning(f"Using total return as CAGR for single day period")
            elif years < 0.1:  # Less than ~25 trading days
                # Use simple annualized return for very short periods
                if mean_return != 0:
                    cagr = mean_return * 252 * 100
                else:
                    cagr = 0.0
                logger.warning(f"Using simple annualized return due to short period ({trading_days} days)")
            else:
                # Standard compound growth calculation
                final_value = np.prod(1 + returns_array)
                if final_value > 0 and years > 0:
                    cagr = (final_value ** (1/years) - 1) * 100
                else:
                    cagr = 0.0
            
            # Cap CAGR at more reasonable bounds
            cagr = max(-99, min(999, cagr))
            
            # Sharpe ratio - handle zero volatility
            excess_returns = returns_array - self.daily_risk_free_rate
            if volatility > 0 and len(returns_array) > 1:
                sharpe_ratio_daily = np.mean(excess_returns) / (np.std(returns_array, ddof=1))
                sharpe_ratio_annual = sharpe_ratio_daily * np.sqrt(252)
            else:
                sharpe_ratio_daily = 0.0
                sharpe_ratio_annual = 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 1:
                downside_deviation = np.std(downside_returns, ddof=1)
                sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0
            else:
                sortino_ratio = 0.0
            
            # Maximum drawdown - improved for small samples
            if len(returns_array) <= 1:
                max_drawdown = 0.0
                max_drawdown_duration = 0
            else:
                cumulative_returns = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                
                max_drawdown = np.min(drawdowns) * 100
                max_drawdown_duration = self._calculate_drawdown_duration(drawdowns)
                
                # Ensure max drawdown is negative or zero
                max_drawdown = min(0.0, max_drawdown)
            
            # Risk-reward ratio
            if volatility > 0:
                risk_reward_ratio = (mean_return * 252) / (volatility / 100)
            else:
                risk_reward_ratio = 0.0
            
            # Win rate and average win/loss
            winning_days = returns_array[returns_array > 0]
            losing_days = returns_array[returns_array < 0]
            
            win_rate = len(winning_days) / len(returns_array) * 100 if len(returns_array) > 0 else 0
            avg_win = np.mean(winning_days) * 100 if len(winning_days) > 0 else 0
            avg_loss = np.mean(losing_days) * 100 if len(losing_days) > 0 else 0
            
            # Calmar ratio (CAGR / Max Drawdown) - handle division by zero
            if abs(max_drawdown) > 0.01:  # At least 0.01% drawdown
                calmar_ratio = abs(cagr / max_drawdown)
            else:
                calmar_ratio = 0.0
            
            metrics = {
                "total_return": float(total_return),
                "cagr": float(cagr),
                "volatility": float(volatility),
                "sharpe_ratio_daily": float(sharpe_ratio_daily),
                "sharpe_ratio_annual": float(sharpe_ratio_annual),
                "sortino_ratio": float(sortino_ratio),
                "max_drawdown": {
                    "max_drawdown": float(max_drawdown),
                    "drawdown_duration": int(max_drawdown_duration)
                },
                "risk_reward_ratio": float(risk_reward_ratio),
                "calmar_ratio": float(calmar_ratio),
                "win_rate": float(win_rate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                "trading_days": int(trading_days),
                "years": float(years),
                "winning_days": int(len(winning_days)),
                "losing_days": int(len(losing_days))
            }
            
            logger.debug(f"Portfolio performance calculated: Return={total_return:.2f}%, Sharpe={sharpe_ratio_annual:.2f}, MaxDD={max_drawdown:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio performance: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._empty_portfolio_metrics()
    
    def _calculate_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """Calculate maximum drawdown duration in days."""
        try:
            max_duration = 0
            current_duration = 0
            
            for dd in drawdowns:
                if dd < 0:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            
            return max_duration
            
        except Exception:
            return 0
    
    def _empty_portfolio_metrics(self) -> Dict[str, Any]:
        """Return empty portfolio metrics structure."""
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe_ratio_daily": 0.0,
            "sharpe_ratio_annual": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": {
                "max_drawdown": 0.0,
                "drawdown_duration": 0
            },
            "risk_reward_ratio": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "trading_days": 0,
            "years": 0.0,
            "winning_days": 0,
            "losing_days": 0
        }
    
    def calculate_risk_metrics(self, portfolio_returns: List[float],
                             benchmark_returns: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate additional risk metrics.
        
        Args:
            portfolio_returns (List[float]): Portfolio returns
            benchmark_returns (Optional[List[float]]): Benchmark returns
            
        Returns:
            Dict[str, float]: Risk metrics
        """
        try:
            if len(portfolio_returns) == 0:
                return {"var_95": 0.0, "cvar_95": 0.0, "skewness": 0.0, "kurtosis": 0.0}
            
            returns_array = np.array(portfolio_returns)
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns_array, 5) * 100
            
            # Conditional Value at Risk (Expected Shortfall)
            cvar_95 = np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)]) * 100
            
            # Skewness and Kurtosis
            skewness = stats.skew(returns_array)
            kurtosis = stats.kurtosis(returns_array)
            
            # Beta and Alpha (if benchmark provided)
            beta = 0.0
            alpha = 0.0
            if benchmark_returns and len(benchmark_returns) > 0:
                bench_array = np.array(benchmark_returns)
                min_length = min(len(returns_array), len(bench_array))
                port_returns = returns_array[:min_length]
                bench_returns = bench_array[:min_length]
                
                if np.std(bench_returns) > 0:
                    beta = np.cov(port_returns, bench_returns)[0, 1] / np.var(bench_returns)
                    alpha = np.mean(port_returns) - beta * np.mean(bench_returns)
                    alpha = alpha * 252 * 100  # Annualized alpha in percentage
            
            metrics = {
                "var_95": float(var_95),
                "cvar_95": float(cvar_95),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "beta": float(beta),
                "alpha_annual": float(alpha)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return {"var_95": 0.0, "cvar_95": 0.0, "skewness": 0.0, "kurtosis": 0.0}
    
    def calculate_all_metrics(self, predicted_returns: List[float],
                            actual_returns: List[float],
                            portfolio_returns: List[float],
                            benchmark_returns: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Calculate all performance metrics in one call.
        
        Args:
            predicted_returns (List[float]): Model predictions
            actual_returns (List[float]): Actual returns
            portfolio_returns (List[float]): Portfolio returns
            benchmark_returns (Optional[List[float]]): Benchmark returns
            
        Returns:
            Dict[str, Any]: Complete metrics dictionary
        """
        try:
            logger.info("📊 Calculating comprehensive performance metrics...")
            
            # Calculate individual metric groups
            return_accuracy = self.calculate_return_accuracy(predicted_returns, actual_returns)
            directional_accuracy = self.calculate_directional_accuracy(predicted_returns, actual_returns)
            portfolio_performance = self.calculate_portfolio_performance(portfolio_returns)
            risk_metrics = self.calculate_risk_metrics(portfolio_returns, benchmark_returns)
            
            # Combine all metrics
            all_metrics = {
                "return_accuracy": return_accuracy,
                "directional_accuracy": directional_accuracy,
                "portfolio_performance": portfolio_performance,
                "risk_metrics": risk_metrics,
                "metadata": {
                    "risk_free_rate": self.risk_free_rate,
                    "calculation_timestamp": datetime.now().isoformat(),
                    "metrics_version": "1.0"
                }
            }
            
            # Log summary
            logger.info(f"✅ Performance metrics calculated:")
            logger.info(f"  Return Accuracy: {return_accuracy.get('accuracy_score', 0):.1f}%")
            logger.info(f"  Directional Accuracy: {directional_accuracy.get('overall_accuracy', 0):.1f}%")
            logger.info(f"  Portfolio Return: {portfolio_performance.get('total_return', 0):.2f}%")
            logger.info(f"  Sharpe Ratio: {portfolio_performance.get('sharpe_ratio_annual', 0):.2f}")
            logger.info(f"  Max Drawdown: {portfolio_performance.get('max_drawdown', {}).get('max_drawdown', 0):.2f}%")
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate all metrics: {e}")
            raise
    
    def create_performance_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a concise performance summary for reporting.
        
        Args:
            metrics (Dict[str, Any]): Complete metrics dictionary
            
        Returns:
            Dict[str, Any]: Performance summary
        """
        try:
            portfolio_perf = metrics.get("portfolio_performance", {})
            return_acc = metrics.get("return_accuracy", {})
            dir_acc = metrics.get("directional_accuracy", {})
            risk_metrics = metrics.get("risk_metrics", {})
            
            summary = {
                "overall_score": self._calculate_overall_score(metrics),
                "key_metrics": {
                    "total_return_pct": portfolio_perf.get("total_return", 0),
                    "sharpe_ratio": portfolio_perf.get("sharpe_ratio_annual", 0),
                    "max_drawdown_pct": portfolio_perf.get("max_drawdown", {}).get("max_drawdown", 0),
                    "return_accuracy_pct": return_acc.get("accuracy_score", 0),
                    "directional_accuracy_pct": dir_acc.get("overall_accuracy", 0),
                    "volatility_pct": portfolio_perf.get("volatility", 0)
                },
                "risk_assessment": self._assess_risk_level(metrics),
                "performance_grade": self._grade_performance(metrics)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create performance summary: {e}")
            return {"overall_score": 0, "key_metrics": {}, "risk_assessment": "Unknown", "performance_grade": "F"}
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)."""
        try:
            portfolio_perf = metrics.get("portfolio_performance", {})
            return_acc = metrics.get("return_accuracy", {})
            dir_acc = metrics.get("directional_accuracy", {})
            
            # Weighted scoring
            sharpe_score = min(portfolio_perf.get("sharpe_ratio_annual", 0) * 25, 25)  # Max 25 points
            return_score = min(return_acc.get("accuracy_score", 0) * 0.25, 25)  # Max 25 points
            direction_score = min(dir_acc.get("overall_accuracy", 0) * 0.25, 25)  # Max 25 points
            
            # Risk penalty
            max_dd = abs(portfolio_perf.get("max_drawdown", {}).get("max_drawdown", 0))
            risk_penalty = min(max_dd * 0.5, 25)  # Max 25 point penalty
            
            total_score = max(0, sharpe_score + return_score + direction_score - risk_penalty)
            return float(total_score)
            
        except Exception:
            return 0.0
    
    def _assess_risk_level(self, metrics: Dict[str, Any]) -> str:
        """Assess overall risk level."""
        try:
            portfolio_perf = metrics.get("portfolio_performance", {})
            volatility = portfolio_perf.get("volatility", 0)
            max_dd = abs(portfolio_perf.get("max_drawdown", {}).get("max_drawdown", 0))
            
            if max_dd > 20 or volatility > 30:
                return "High Risk"
            elif max_dd > 10 or volatility > 20:
                return "Medium Risk"
            else:
                return "Low Risk"
                
        except Exception:
            return "Unknown"
    
    def _grade_performance(self, metrics: Dict[str, Any]) -> str:
        """Grade overall performance A-F."""
        try:
            score = self._calculate_overall_score(metrics)
            
            if score >= 80:
                return "A"
            elif score >= 70:
                return "B"
            elif score >= 60:
                return "C"
            elif score >= 50:
                return "D"
            else:
                return "F"
                
        except Exception:
            return "F"

# Utility functions for easy access
def calculate_portfolio_metrics(portfolio_returns: List[float], 
                               risk_free_rate: float = 0.06) -> Dict[str, Any]:
    """
    Quick function to calculate portfolio metrics.
    
    Args:
        portfolio_returns (List[float]): Daily portfolio returns
        risk_free_rate (float): Annual risk-free rate
        
    Returns:
        Dict[str, Any]: Portfolio performance metrics
    """
    calculator = PerformanceMetrics(risk_free_rate)
    return calculator.calculate_portfolio_performance(portfolio_returns)

def calculate_model_accuracy(predicted_returns: List[float],
                           actual_returns: List[float]) -> Dict[str, float]:
    """
    Quick function to calculate model accuracy.
    
    Args:
        predicted_returns (List[float]): Model predictions
        actual_returns (List[float]): Actual returns
        
    Returns:
        Dict[str, float]: Accuracy metrics
    """
    calculator = PerformanceMetrics()
    return_acc = calculator.calculate_return_accuracy(predicted_returns, actual_returns)
    dir_acc = calculator.calculate_directional_accuracy(predicted_returns, actual_returns)
    
    return {
        **return_acc,
        **dir_acc
    }

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example performance calculation
    try:
        # Create sample data
        np.random.seed(42)
        n_days = 100
        
        predicted_returns = np.random.normal(0.001, 0.02, n_days)
        actual_returns = predicted_returns + np.random.normal(0, 0.01, n_days)
        portfolio_returns = actual_returns * 0.8  # Simulated portfolio performance
        
        # Calculate metrics
        calculator = PerformanceMetrics(risk_free_rate=0.06)
        all_metrics = calculator.calculate_all_metrics(
            predicted_returns.tolist(),
            actual_returns.tolist(),
            portfolio_returns.tolist()
        )
        
        # Create summary
        summary = calculator.create_performance_summary(all_metrics)
        
        print("Performance metrics calculation example completed successfully!")
        print(f"Overall Score: {summary['overall_score']:.1f}/100")
        print(f"Performance Grade: {summary['performance_grade']}")
        print(f"Risk Assessment: {summary['risk_assessment']}")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc() 