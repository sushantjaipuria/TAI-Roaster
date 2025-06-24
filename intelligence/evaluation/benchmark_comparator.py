"""
Benchmark Comparator - Portfolio vs Market Benchmark Analysis
Compares portfolio performance against various market benchmarks as specified in PRD.

Benchmarks supported:
- NIFTY50 (Primary benchmark)
- NIFTY100, NIFTY500 (Additional benchmarks)
- Sector-specific indices
- Risk-free rate comparison
- Custom benchmark support
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import yfinance as yf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class BenchmarkComparator:
    """
    Comprehensive benchmark comparison engine for portfolio evaluation.
    
    Features:
    - NIFTY50 and other Indian market indices
    - Beta, Alpha, and tracking error calculation
    - Information ratio and Treynor ratio
    - Up/down market analysis
    - Correlation analysis
    - Risk-adjusted performance comparison
    """
    
    def __init__(self, primary_benchmark: str = "^NSEI"):
        """
        Initialize benchmark comparator.
        
        Args:
            primary_benchmark (str): Primary benchmark ticker (default: NIFTY50)
        """
        self.primary_benchmark = primary_benchmark
        self.available_benchmarks = {
            "^NSEI": "NIFTY 50",
            "^NSEBANK": "NIFTY Bank",
            "^NSEIT": "NIFTY IT", 
            "^NSEAUTO": "NIFTY Auto",
            "^NSEPHARMA": "NIFTY Pharma",
            "^NSEI100": "NIFTY 100",
            "^NSEI500": "NIFTY 500"
        }
        
        logger.info(f"BenchmarkComparator initialized with primary benchmark: {self.available_benchmarks.get(primary_benchmark, primary_benchmark)}")
    
    def fetch_benchmark_data(self, benchmark_ticker: str, 
                           start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch benchmark data for comparison.
        
        Args:
            benchmark_ticker (str): Benchmark ticker symbol
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Benchmark price and return data
        """
        try:
            logger.info(f"📊 Fetching benchmark data for {self.available_benchmarks.get(benchmark_ticker, benchmark_ticker)}")
            
            # Fetch data using yfinance
            benchmark = yf.Ticker(benchmark_ticker)
            data = benchmark.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for benchmark {benchmark_ticker}")
                return pd.DataFrame()
            
            # Process benchmark data
            benchmark_data = pd.DataFrame({
                'date': data.index,
                'close': data['Close'],
                'volume': data['Volume']
            }).reset_index(drop=True)
            
            # Calculate returns
            benchmark_data['daily_return'] = benchmark_data['close'].pct_change()
            benchmark_data['cumulative_return'] = (1 + benchmark_data['daily_return']).cumprod() - 1
            
            # Fill NaN values
            benchmark_data = benchmark_data.fillna(0)
            
            logger.info(f"✅ Fetched {len(benchmark_data)} days of benchmark data")
            return benchmark_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch benchmark data for {benchmark_ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_benchmark_metrics(self, portfolio_returns: List[float],
                                  benchmark_returns: List[float],
                                  risk_free_rate: float = 0.06) -> Dict[str, float]:
        """
        Calculate comprehensive benchmark comparison metrics.
        
        Args:
            portfolio_returns (List[float]): Portfolio daily returns
            benchmark_returns (List[float]): Benchmark daily returns
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            Dict[str, float]: Benchmark comparison metrics
        """
        try:
            if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
                logger.warning("Empty returns data for benchmark comparison")
                return self._empty_benchmark_metrics()
            
            port_array = np.array(portfolio_returns)
            bench_array = np.array(benchmark_returns)
            
            # Align data to same length
            min_length = min(len(port_array), len(bench_array))
            port_returns = port_array[:min_length]
            bench_returns = bench_array[:min_length]
            
            daily_rf_rate = risk_free_rate / 252
            
            # Basic statistics
            port_mean = np.mean(port_returns)
            bench_mean = np.mean(bench_returns)
            port_std = np.std(port_returns)
            bench_std = np.std(bench_returns)
            
            # Beta calculation
            if bench_std > 0:
                covariance = np.cov(port_returns, bench_returns)[0, 1]
                beta = covariance / np.var(bench_returns)
            else:
                beta = 0.0
            
            # Alpha calculation (CAPM)
            alpha_daily = port_mean - (daily_rf_rate + beta * (bench_mean - daily_rf_rate))
            alpha_annual = alpha_daily * 252 * 100
            
            # Correlation
            correlation = np.corrcoef(port_returns, bench_returns)[0, 1] if port_std > 0 and bench_std > 0 else 0.0
            
            # Information Ratio
            excess_returns = port_returns - bench_returns
            tracking_error = np.std(excess_returns)
            information_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0.0
            
            # Treynor Ratio
            treynor_portfolio = (port_mean - daily_rf_rate) / beta if beta != 0 else 0.0
            treynor_benchmark = (bench_mean - daily_rf_rate) / 1.0  # Beta of benchmark is 1
            
            # Up/Down market analysis
            up_market_mask = bench_returns > 0
            down_market_mask = bench_returns < 0
            
            up_capture = 0.0
            down_capture = 0.0
            
            if np.sum(up_market_mask) > 0:
                port_up_return = np.mean(port_returns[up_market_mask])
                bench_up_return = np.mean(bench_returns[up_market_mask])
                up_capture = (port_up_return / bench_up_return) if bench_up_return != 0 else 0.0
            
            if np.sum(down_market_mask) > 0:
                port_down_return = np.mean(port_returns[down_market_mask])
                bench_down_return = np.mean(bench_returns[down_market_mask])
                down_capture = (port_down_return / bench_down_return) if bench_down_return != 0 else 0.0
            
            # R-squared
            r_squared = correlation ** 2
            
            # Excess return
            excess_return_annual = (port_mean - bench_mean) * 252 * 100
            
            metrics = {
                "beta": float(beta),
                "alpha_annual": float(alpha_annual),
                "correlation": float(correlation),
                "r_squared": float(r_squared),
                "information_ratio": float(information_ratio),
                "tracking_error": float(tracking_error * np.sqrt(252) * 100),  # Annualized %
                "treynor_ratio_portfolio": float(treynor_portfolio),
                "treynor_ratio_benchmark": float(treynor_benchmark),
                "up_capture_ratio": float(up_capture),
                "down_capture_ratio": float(down_capture),
                "excess_return_annual": float(excess_return_annual),
                "up_market_periods": int(np.sum(up_market_mask)),
                "down_market_periods": int(np.sum(down_market_mask)),
                "sample_size": int(min_length)
            }
            
            logger.debug(f"Benchmark metrics calculated: Beta={beta:.2f}, Alpha={alpha_annual:.2f}%, Info Ratio={information_ratio:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate benchmark metrics: {e}")
            return self._empty_benchmark_metrics()
    
    def _empty_benchmark_metrics(self) -> Dict[str, float]:
        """Return empty benchmark metrics structure."""
        return {
            "beta": 0.0,
            "alpha_annual": 0.0,
            "correlation": 0.0,
            "r_squared": 0.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
            "treynor_ratio_portfolio": 0.0,
            "treynor_ratio_benchmark": 0.0,
            "up_capture_ratio": 0.0,
            "down_capture_ratio": 0.0,
            "excess_return_annual": 0.0,
            "up_market_periods": 0,
            "down_market_periods": 0,
            "sample_size": 0
        }
    
    def create_benchmark_comparison_report(self, portfolio_returns: List[float],
                                         start_date: str, end_date: str,
                                         benchmarks: Optional[List[str]] = None,
                                         risk_free_rate: float = 0.06) -> Dict[str, Any]:
        """
        Create comprehensive benchmark comparison report.
        
        Args:
            portfolio_returns (List[float]): Portfolio daily returns
            start_date (str): Start date for benchmark data
            end_date (str): End date for benchmark data
            benchmarks (Optional[List[str]]): List of benchmark tickers
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            Dict[str, Any]: Complete benchmark comparison report
        """
        try:
            logger.info("🏆 Creating comprehensive benchmark comparison report...")
            
            if benchmarks is None:
                benchmarks = [self.primary_benchmark]
            
            all_benchmark_data = {}
            benchmark_comparisons = {}
            
            # Fetch and compare against each benchmark
            for benchmark_ticker in benchmarks:
                try:
                    # Fetch benchmark data
                    benchmark_data = self.fetch_benchmark_data(benchmark_ticker, start_date, end_date)
                    
                    if benchmark_data.empty:
                        logger.warning(f"Skipping {benchmark_ticker} - no data available")
                        continue
                    
                    # Calculate comparison metrics
                    benchmark_returns = benchmark_data['daily_return'].tolist()
                    comparison_metrics = self.calculate_benchmark_metrics(
                        portfolio_returns, benchmark_returns, risk_free_rate
                    )
                    
                    # Store results
                    benchmark_name = self.available_benchmarks.get(benchmark_ticker, benchmark_ticker)
                    all_benchmark_data[benchmark_ticker] = {
                        "name": benchmark_name,
                        "data": benchmark_data,
                        "returns": benchmark_returns
                    }
                    
                    benchmark_comparisons[benchmark_ticker] = {
                        "name": benchmark_name,
                        "metrics": comparison_metrics
                    }
                    
                    logger.info(f"✅ Completed comparison with {benchmark_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process benchmark {benchmark_ticker}: {e}")
                    continue
            
            # Create summary analysis
            summary = self._create_benchmark_summary(benchmark_comparisons, portfolio_returns)
            
            # Calculate portfolio performance metrics for comparison
            portfolio_stats = self._calculate_portfolio_stats(portfolio_returns, risk_free_rate)
            
            report = {
                "summary": summary,
                "portfolio_stats": portfolio_stats,
                "benchmark_comparisons": benchmark_comparisons,
                "benchmark_data": {k: {"name": v["name"], "sample_size": len(v["data"])} 
                                 for k, v in all_benchmark_data.items()},
                "analysis_metadata": {
                    "primary_benchmark": self.primary_benchmark,
                    "benchmarks_analyzed": len(benchmark_comparisons),
                    "period": {"start_date": start_date, "end_date": end_date},
                    "risk_free_rate": risk_free_rate,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ Benchmark comparison report completed with {len(benchmark_comparisons)} benchmarks")
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to create benchmark comparison report: {e}")
            raise
    
    def _create_benchmark_summary(self, benchmark_comparisons: Dict[str, Any], 
                                portfolio_returns: List[float]) -> Dict[str, Any]:
        """Create summary analysis from benchmark comparisons."""
        try:
            if not benchmark_comparisons:
                return {
                    "primary_benchmark": "NIFTY 50",
                    "excess_return": 0.0,
                    "information_ratio": 0.0,
                    "alpha_annual": 0.0,
                    "beta": 0.0,
                    "outperformance_status": "No Data",
                    "risk_profile": "Unknown",
                    "tracking_quality": "Unknown",
                    "correlation": 0.0,
                    "error": "No valid benchmark comparisons available"
                }
            
            # Use primary benchmark for main comparison
            primary_comparison = None
            for ticker, data in benchmark_comparisons.items():
                if ticker == self.primary_benchmark:
                    primary_comparison = data
                    break
            
            # If primary not found, use first available
            if primary_comparison is None:
                primary_comparison = next(iter(benchmark_comparisons.values()))
            
            primary_metrics = primary_comparison["metrics"]
            
            # Overall performance assessment
            excess_return = primary_metrics.get("excess_return_annual", 0)
            information_ratio = primary_metrics.get("information_ratio", 0)
            
            outperformance = "Outperforming" if excess_return > 0 else "Underperforming"
            
            # Risk-adjusted performance
            alpha = primary_metrics.get("alpha_annual", 0)
            beta = primary_metrics.get("beta", 1.0)
            
            risk_profile = "Lower Risk" if beta < 1.0 else "Higher Risk" if beta > 1.2 else "Similar Risk"
            
            summary = {
                "primary_benchmark": primary_comparison.get("name", "NIFTY 50"),
                "excess_return": float(excess_return),
                "information_ratio": float(information_ratio),
                "alpha_annual": float(alpha),
                "beta": float(beta),
                "outperformance_status": outperformance,
                "risk_profile": risk_profile,
                "tracking_quality": "High" if primary_metrics.get("r_squared", 0) > 0.8 else 
                                   "Medium" if primary_metrics.get("r_squared", 0) > 0.5 else "Low",
                "correlation": float(primary_metrics.get("correlation", 0))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create benchmark summary: {e}")
            return {
                "primary_benchmark": "NIFTY 50",
                "excess_return": 0.0,
                "information_ratio": 0.0,
                "alpha_annual": 0.0,
                "beta": 0.0,
                "outperformance_status": "Error",
                "risk_profile": "Unknown",
                "tracking_quality": "Unknown",
                "correlation": 0.0,
                "error": str(e)
            }
    
    def _calculate_portfolio_stats(self, portfolio_returns: List[float], 
                                 risk_free_rate: float) -> Dict[str, float]:
        """Calculate basic portfolio statistics for comparison."""
        try:
            if len(portfolio_returns) == 0:
                return {}
            
            returns_array = np.array(portfolio_returns)
            daily_rf_rate = risk_free_rate / 252
            
            total_return = (np.prod(1 + returns_array) - 1) * 100
            volatility = np.std(returns_array) * np.sqrt(252) * 100
            sharpe_ratio = (np.mean(returns_array) - daily_rf_rate) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            
            return {
                "total_return_pct": float(total_return),
                "volatility_pct": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "trading_days": len(returns_array)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio stats: {e}")
            return {}

# Utility functions
def compare_with_nifty50(portfolio_returns: List[float], 
                        start_date: str, end_date: str,
                        risk_free_rate: float = 0.06) -> Dict[str, Any]:
    """
    Quick function to compare portfolio with NIFTY50.
    
    Args:
        portfolio_returns (List[float]): Portfolio daily returns
        start_date (str): Start date
        end_date (str): End date
        risk_free_rate (float): Risk-free rate
        
    Returns:
        Dict[str, Any]: NIFTY50 comparison results
    """
    comparator = BenchmarkComparator()
    return comparator.create_benchmark_comparison_report(
        portfolio_returns, start_date, end_date, ["^NSEI"], risk_free_rate
    )

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example benchmark comparison
    try:
        # Create sample portfolio returns
        np.random.seed(42)
        n_days = 100
        portfolio_returns = np.random.normal(0.001, 0.02, n_days).tolist()
        
        start_date = "2024-01-01"
        end_date = "2024-06-30"
        
        # Compare with NIFTY50
        comparator = BenchmarkComparator()
        report = comparator.create_benchmark_comparison_report(
            portfolio_returns, start_date, end_date
        )
        
        print("Benchmark comparison example completed successfully!")
        print(f"Primary benchmark: {report['summary'].get('primary_benchmark', 'N/A')}")
        print(f"Excess return: {report['summary'].get('excess_return', 0):.2f}%")
        print(f"Information ratio: {report['summary'].get('information_ratio', 0):.3f}")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc() 