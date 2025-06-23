"""
Results Logger - JSON & Markdown Report Writer
Generates comprehensive evaluation reports in multiple formats.

Requirements from PRD:
- JSON, Markdown, and PDF report generation
- Daily logs stored in structured folders
- Readable performance reports with visualizations
- API consumable report metadata
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
logger = logging.getLogger(__name__)

class ResultsLogger:
    """
    Handles comprehensive logging and report generation for evaluation results.
    
    Outputs:
    - JSON: Machine-readable metrics and metadata
    - Markdown: Human-readable reports
    - PDF: Formatted reports (via markdown)
    - PNG: Visualization charts
    """
    
    def __init__(self, base_output_dir: str = "logs/evaluation"):
        """
        Initialize results logger.
        
        Args:
            base_output_dir (str): Base directory for all evaluation logs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_daily_folder(self, date: str = None) -> Path:
        """
        Create folder structure for daily evaluation results.
        
        Args:
            date (str): Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Path: Path to daily folder
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        daily_folder = self.base_output_dir / date
        daily_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created daily evaluation folder: {daily_folder}")
        return daily_folder
    
    def save_json_report(self, metrics: Dict, output_dir: Path, 
                        filename: str = "evaluation_summary.json") -> Path:
        """
        Save evaluation metrics as JSON.
        
        Args:
            metrics (Dict): Complete evaluation metrics
            output_dir (Path): Output directory
            filename (str): JSON filename
            
        Returns:
            Path: Path to saved JSON file
        """
        json_path = output_dir / filename
        
        # Add metadata
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "evaluation_summary",
                "version": "1.0"
            },
            "evaluation_results": metrics
        }
        
        try:
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"✅ JSON report saved: {json_path}")
            return json_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save JSON report: {e}")
            raise
    
    def save_markdown_report(self, metrics: Dict, output_dir: Path,
                           filename: str = "evaluation_report.md") -> Path:
        """
        Generate comprehensive Markdown report.
        
        Args:
            metrics (Dict): Complete evaluation metrics
            output_dir (Path): Output directory
            filename (str): Markdown filename
            
        Returns:
            Path: Path to saved Markdown file
        """
        md_path = output_dir / filename
        
        try:
            markdown_content = self._generate_markdown_content(metrics)
            
            with open(md_path, 'w') as f:
                f.write(markdown_content)
            
            logger.info(f"✅ Markdown report saved: {md_path}")
            return md_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save Markdown report: {e}")
            raise
    
    def generate_visualization_charts(self, metrics: Dict, output_dir: Path) -> List[Path]:
        """
        Generate visualization charts for the evaluation report.
        
        Args:
            metrics (Dict): Evaluation metrics
            output_dir (Path): Output directory
            
        Returns:
            List[Path]: List of generated chart files
        """
        chart_paths = []
        
        try:
            # 1. Performance Summary Chart
            summary_path = self._create_performance_summary_chart(metrics, output_dir)
            if summary_path:
                chart_paths.append(summary_path)
            
            # 2. Return Distribution Chart
            distribution_path = self._create_return_distribution_chart(metrics, output_dir)
            if distribution_path:
                chart_paths.append(distribution_path)
            
            # 3. Benchmark Comparison Chart
            benchmark_path = self._create_benchmark_comparison_chart(metrics, output_dir)
            if benchmark_path:
                chart_paths.append(benchmark_path)
            
            # 4. Feature Importance Chart (if SHAP data available)
            shap_path = self._create_feature_importance_chart(metrics, output_dir)
            if shap_path:
                chart_paths.append(shap_path)
            
            logger.info(f"✅ Generated {len(chart_paths)} visualization charts")
            return chart_paths
            
        except Exception as e:
            logger.error(f"❌ Failed to generate charts: {e}")
            return chart_paths
    
    def create_complete_report(self, metrics: Dict, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Create complete evaluation report with all formats.
        
        Args:
            metrics (Dict): Evaluation metrics
            output_dir (Optional[Path]): Output directory (creates daily folder if None)
            
        Returns:
            Dict[str, Path]: Paths to all generated files
        """
        if output_dir is None:
            output_dir = self.create_daily_folder()
        
        generated_files = {}
        
        try:
            # Generate JSON report
            json_path = self.save_json_report(metrics, output_dir)
            generated_files['json'] = json_path
            
            # Generate Markdown report
            markdown_path = self.save_markdown_report(metrics, output_dir)
            generated_files['markdown'] = markdown_path
            
            # Generate visualization charts
            chart_paths = self.generate_visualization_charts(metrics, output_dir)
            generated_files['charts'] = chart_paths
            
            # Generate PDF from Markdown (if pandoc available)
            try:
                pdf_path = self._convert_markdown_to_pdf(markdown_path, output_dir)
                if pdf_path:
                    generated_files['pdf'] = pdf_path
            except Exception as e:
                logger.warning(f"PDF generation failed: {e}")
            
            # Create summary metadata
            summary_path = self._create_report_summary(generated_files, output_dir)
            generated_files['summary'] = summary_path
            
            logger.info(f"✅ Complete evaluation report generated in: {output_dir}")
            return generated_files
            
        except Exception as e:
            logger.error(f"❌ Failed to create complete report: {e}")
            raise
    
    def _generate_markdown_content(self, metrics: Dict) -> str:
        """Generate comprehensive Markdown content."""
        content = []
        
        # Header
        content.append("# TAI Trading System - Evaluation Report")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # Executive Summary
        content.append("## 📊 Executive Summary")
        
        if "portfolio_performance" in metrics:
            perf = metrics["portfolio_performance"]
            content.append(f"- **Total Return:** {perf.get('total_return', 0):.2f}%")
            content.append(f"- **CAGR:** {perf.get('cagr', 0):.2f}%")
            content.append(f"- **Sharpe Ratio:** {perf.get('sharpe_ratio_daily', 0):.2f}")
            content.append(f"- **Max Drawdown:** {perf.get('max_drawdown', {}).get('max_drawdown', 0):.2f}%")
            content.append(f"- **Volatility:** {perf.get('volatility', 0):.2f}%")
        
        content.append("")
        
        # Model Accuracy
        content.append("## 🎯 Model Accuracy")
        
        if "return_accuracy" in metrics:
            acc = metrics["return_accuracy"]
            content.append(f"- **Return Prediction Accuracy:** {acc.get('accuracy_score', 0):.2f}%")
            content.append(f"- **Correlation:** {acc.get('correlation', 0):.3f}")
            content.append(f"- **RMSE:** {acc.get('rmse', 0):.4f}")
        
        if "directional_accuracy" in metrics:
            dir_acc = metrics["directional_accuracy"]
            content.append(f"- **Directional Accuracy:** {dir_acc.get('overall_accuracy', 0):.2f}%")
            content.append(f"- **Up Movement Accuracy:** {dir_acc.get('up_movement_accuracy', 0):.2f}%")
            content.append(f"- **Down Movement Accuracy:** {dir_acc.get('down_movement_accuracy', 0):.2f}%")
        
        content.append("")
        
        # Benchmark Comparison
        if "benchmark_comparison" in metrics:
            content.append("## 📈 Benchmark Comparison")
            bench = metrics["benchmark_comparison"]
            content.append(f"- **Excess Return vs Benchmark:** {bench.get('excess_return', 0):.2f}%")
            content.append(f"- **Information Ratio:** {bench.get('information_ratio', 0):.2f}")
            content.append(f"- **Beta:** {bench.get('beta', 1.0):.2f}")
            content.append(f"- **Alpha (Annualized):** {bench.get('alpha_annual', 0):.2f}%")
            content.append("")
        
        # Risk Metrics
        content.append("## ⚠️ Risk Analysis")
        if "portfolio_performance" in metrics:
            perf = metrics["portfolio_performance"]
            content.append(f"- **Risk-Reward Ratio:** {perf.get('risk_reward_ratio', 0):.2f}")
            content.append(f"- **Maximum Drawdown:** {perf.get('max_drawdown', {}).get('max_drawdown', 0):.2f}%")
            content.append(f"- **Drawdown Duration:** {perf.get('max_drawdown', {}).get('drawdown_duration', 0)} days")
        
        content.append("")
        
        # Data Quality
        content.append("## 📋 Data Quality")
        content.append(f"- **Evaluation Period:** {metrics.get('data_points', 0)} trading days")
        content.append(f"- **Report Timestamp:** {metrics.get('timestamp', 'N/A')}")
        
        # Feature Importance (if available)
        if "feature_importance" in metrics:
            content.append("")
            content.append("## 🔍 Top Features")
            features = metrics["feature_importance"]
            for i, (feature, importance) in enumerate(features[:10], 1):
                content.append(f"{i}. **{feature}:** {importance:.4f}")
        
        # Visualizations
        content.append("")
        content.append("## 📊 Visualizations")
        content.append("The following charts are available in the report folder:")
        content.append("- Performance Summary Chart")
        content.append("- Return Distribution Analysis")
        content.append("- Benchmark Comparison")
        content.append("- Feature Importance (if available)")
        
        # Footer
        content.append("")
        content.append("---")
        content.append("*Generated by TAI Trading System Evaluation Engine*")
        
        return "\n".join(content)
    
    def _create_performance_summary_chart(self, metrics: Dict, output_dir: Path) -> Optional[Path]:
        """Create performance summary visualization."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('TAI Trading System - Performance Summary', fontsize=16, fontweight='bold')
            
            # Chart 1: Key Metrics Bar Chart
            if "portfolio_performance" in metrics:
                perf = metrics["portfolio_performance"]
                metrics_names = ['Total Return (%)', 'CAGR (%)', 'Sharpe Ratio', 'Max DD (%)']
                metrics_values = [
                    perf.get('total_return', 0),
                    perf.get('cagr', 0),
                    perf.get('sharpe_ratio_daily', 0),
                    -perf.get('max_drawdown', {}).get('max_drawdown', 0)
                ]
                
                colors = ['green' if v >= 0 else 'red' for v in metrics_values]
                ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
                ax1.set_title('Key Performance Metrics')
                ax1.tick_params(axis='x', rotation=45)
            
            # Chart 2: Accuracy Metrics
            if "return_accuracy" in metrics and "directional_accuracy" in metrics:
                acc_names = ['Return Accuracy (%)', 'Directional Accuracy (%)']
                acc_values = [
                    metrics["return_accuracy"].get('accuracy_score', 0),
                    metrics["directional_accuracy"].get('overall_accuracy', 0)
                ]
                
                ax2.bar(acc_names, acc_values, color=['blue', 'orange'], alpha=0.7)
                ax2.set_title('Model Accuracy')
                ax2.set_ylim(0, 100)
                ax2.tick_params(axis='x', rotation=45)
            
            # Chart 3: Risk-Return Scatter
            if "portfolio_performance" in metrics and "benchmark_comparison" in metrics:
                portfolio_return = metrics["portfolio_performance"].get('total_return', 0)
                portfolio_vol = metrics["portfolio_performance"].get('volatility', 0)
                benchmark_return = metrics["benchmark_comparison"].get('benchmark_total_return', 0)
                benchmark_vol = 15.0  # Typical NIFTY volatility
                
                ax3.scatter([portfolio_vol], [portfolio_return], s=100, color='blue', label='Portfolio', alpha=0.7)
                ax3.scatter([benchmark_vol], [benchmark_return], s=100, color='red', label='Benchmark', alpha=0.7)
                ax3.set_xlabel('Volatility (%)')
                ax3.set_ylabel('Return (%)')
                ax3.set_title('Risk-Return Profile')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Chart 4: Performance Attribution
            if "benchmark_comparison" in metrics:
                bench = metrics["benchmark_comparison"]
                attribution = [
                    bench.get('excess_return', 0),
                    bench.get('alpha_annual', 0)
                ]
                attr_names = ['Excess Return (%)', 'Alpha (%)']
                
                colors = ['green' if v >= 0 else 'red' for v in attribution]
                ax4.bar(attr_names, attribution, color=colors, alpha=0.7)
                ax4.set_title('Performance Attribution')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = output_dir / "performance_summary.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to create performance summary chart: {e}")
            return None
    
    def _create_return_distribution_chart(self, metrics: Dict, output_dir: Path) -> Optional[Path]:
        """Create return distribution analysis."""
        try:
            # This would need actual return data - placeholder for now
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Return Distribution Analysis', fontsize=14, fontweight='bold')
            
            # Placeholder - would use actual return data
            returns = np.random.normal(0.001, 0.02, 100)  # Dummy data
            
            # Histogram
            ax1.hist(returns * 100, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('Daily Returns (%)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Return Distribution')
            ax1.axvline(x=np.mean(returns) * 100, color='red', linestyle='--', label='Mean')
            ax1.legend()
            
            # Q-Q Plot
            from scipy import stats
            stats.probplot(returns, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normal Distribution)')
            
            plt.tight_layout()
            
            chart_path = output_dir / "return_distribution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to create return distribution chart: {e}")
            return None
    
    def _create_benchmark_comparison_chart(self, metrics: Dict, output_dir: Path) -> Optional[Path]:
        """Create benchmark comparison visualization."""
        try:
            if "benchmark_comparison" not in metrics:
                return None
            
            bench = metrics["benchmark_comparison"]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Benchmark Comparison', fontsize=14, fontweight='bold')
            
            # Performance comparison
            categories = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
            portfolio_values = [
                metrics.get("portfolio_performance", {}).get('total_return', 0),
                metrics.get("portfolio_performance", {}).get('sharpe_ratio_daily', 0),
                -metrics.get("portfolio_performance", {}).get('max_drawdown', {}).get('max_drawdown', 0)
            ]
            benchmark_values = [
                bench.get('benchmark_total_return', 0),
                bench.get('benchmark_sharpe', 0),
                -bench.get('benchmark_max_dd', {}).get('max_drawdown', 0) if isinstance(bench.get('benchmark_max_dd'), dict) else 0
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax1.bar(x - width/2, portfolio_values, width, label='Portfolio', alpha=0.7)
            ax1.bar(x + width/2, benchmark_values, width, label='Benchmark', alpha=0.7)
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.set_title('Performance Comparison')
            ax1.legend()
            
            # Risk-adjusted metrics
            risk_metrics = ['Information Ratio', 'Alpha (%)', 'Beta']
            risk_values = [
                bench.get('information_ratio', 0),
                bench.get('alpha_annual', 0),
                bench.get('beta', 1.0)
            ]
            
            colors = ['green' if v >= 0 else 'red' for v in risk_values]
            ax2.bar(risk_metrics, risk_values, color=colors, alpha=0.7)
            ax2.set_title('Risk-Adjusted Metrics')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = output_dir / "benchmark_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to create benchmark comparison chart: {e}")
            return None
    
    def _create_feature_importance_chart(self, metrics: Dict, output_dir: Path) -> Optional[Path]:
        """Create feature importance visualization."""
        try:
            if "feature_importance" not in metrics:
                return None
            
            features = metrics["feature_importance"][:10]  # Top 10
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            feature_names = [f[0] for f in features]
            importance_values = [f[1] for f in features]
            
            ax.barh(feature_names, importance_values, alpha=0.7, color='steelblue')
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 10 Feature Importance (SHAP Values)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = output_dir / "feature_importance.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to create feature importance chart: {e}")
            return None
    
    def _convert_markdown_to_pdf(self, markdown_path: Path, output_dir: Path) -> Optional[Path]:
        """Convert Markdown to PDF using pandoc."""
        try:
            import subprocess
            
            pdf_path = output_dir / "evaluation_report.pdf"
            
            # Try to use pandoc
            result = subprocess.run([
                'pandoc', str(markdown_path), '-o', str(pdf_path),
                '--pdf-engine=wkhtmltopdf'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ PDF report generated: {pdf_path}")
                return pdf_path
            else:
                logger.warning(f"PDF generation failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            logger.warning("Pandoc not found - PDF generation skipped")
            return None
        except Exception as e:
            logger.warning(f"PDF generation failed: {e}")
            return None
    
    def _create_report_summary(self, generated_files: Dict[str, Path], output_dir: Path) -> Path:
        """Create a summary of all generated files."""
        summary_path = output_dir / "report_summary.json"
        
        summary = {
            "report_date": datetime.now().isoformat(),
            "output_directory": str(output_dir),
            "generated_files": {
                key: str(path) if isinstance(path, Path) else [str(p) for p in path]
                for key, path in generated_files.items()
            },
            "file_count": sum(1 if isinstance(path, Path) else len(path) for path in generated_files.values())
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary_path

# Example usage
if __name__ == "__main__":
    # Test results logger
    logger_instance = ResultsLogger()
    
    # Sample metrics
    test_metrics = {
        "timestamp": datetime.now().isoformat(),
        "data_points": 100,
        "return_accuracy": {
            "accuracy_score": 75.5,
            "correlation": 0.65,
            "rmse": 0.02
        },
        "directional_accuracy": {
            "overall_accuracy": 68.0,
            "up_movement_accuracy": 72.0,
            "down_movement_accuracy": 64.0
        },
        "portfolio_performance": {
            "total_return": 12.5,
            "cagr": 15.2,
            "sharpe_ratio_daily": 1.25,
            "max_drawdown": {"max_drawdown": 8.5, "drawdown_duration": 15},
            "volatility": 18.5
        },
        "benchmark_comparison": {
            "excess_return": 3.2,
            "information_ratio": 0.85,
            "alpha_annual": 2.1,
            "beta": 0.95
        }
    }
    
    # Generate complete report
    generated_files = logger_instance.create_complete_report(test_metrics)
    print("Generated files:", generated_files) 