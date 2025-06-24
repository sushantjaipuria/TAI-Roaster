# intelligence/transparency_logger.py

import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict

# Custom JSON encoder to handle pandas Timestamps and numpy types
class TransparencyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# Helper function to clean data for JSON serialization
def clean_for_json(obj):
    """Clean object for JSON serialization by converting non-serializable types"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, 'to_dict'):
        return clean_for_json(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return clean_for_json(obj.__dict__)
    else:
        try:
            # Try to convert to a basic type
            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
        except:
            return str(obj)

@dataclass
class CalculationStep:
    """Individual calculation step for transparency"""
    step_name: str
    input_data: Dict[str, Any]
    calculation_method: str
    output_data: Dict[str, Any]
    timestamp: str
    processing_time_ms: float
    notes: str = ""

@dataclass
class ModelOutput:
    """Model prediction output with full details"""
    model_name: str
    model_version: str
    input_features: List[float]
    feature_names: List[str]
    raw_prediction: float
    confidence_score: float
    model_parameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    prediction_explanation: str
    processing_time_ms: float

@dataclass
class LLMAnalysisLog:
    """LLM analysis with full prompt and response"""
    stage: str  # "validation" or "strategy"
    prompt_sent: str
    response_received: str
    parsed_output: Dict[str, Any]
    confidence_level: float
    reasoning: List[str]
    processing_time_ms: float
    api_provider: str
    model_used: str

@dataclass
class StockTransparencyLog:
    """Complete transparency log for a single stock"""
    ticker: str
    analysis_timestamp: str
    user_input: Dict[str, Any]
    
    # Data Collection
    raw_data: Dict[str, Any]
    data_quality_metrics: Dict[str, Any]
    
    # Feature Engineering
    feature_calculations: List[CalculationStep]
    final_features: Dict[str, float]
    feature_statistics: Dict[str, Any]
    
    # Model Predictions
    model_outputs: List[ModelOutput]
    ensemble_calculation: CalculationStep
    
    # LLM Analysis
    llm_analyses: List[LLMAnalysisLog]
    
    # Final Decision
    final_recommendation: Dict[str, Any]
    decision_rationale: str
    risk_assessment: Dict[str, Any]
    
    # Performance Metrics
    total_processing_time_ms: float
    system_performance: Dict[str, Any]

class TransparencyLogger:
    """Main transparency logging system"""
    
    def __init__(self, output_dir: str = "transparency_logs"):
        self.output_dir = output_dir
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_logs: Dict[str, StockTransparencyLog] = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/{self.current_session_id}", exist_ok=True)
    
    def start_stock_analysis(self, ticker: str, user_input: Dict[str, Any]) -> None:
        """Initialize transparency log for a stock"""
        self.session_logs[ticker] = StockTransparencyLog(
            ticker=ticker,
            analysis_timestamp=datetime.now().isoformat(),
            user_input=user_input,
            raw_data={},
            data_quality_metrics={},
            feature_calculations=[],
            final_features={},
            feature_statistics={},
            model_outputs=[],
            ensemble_calculation=None,
            llm_analyses=[],
            final_recommendation={},
            decision_rationale="",
            risk_assessment={},
            total_processing_time_ms=0.0,
            system_performance={}
        )
    
    def log_data_collection(self, ticker: str, raw_data: pd.DataFrame, 
                           quality_metrics: Dict[str, Any]) -> None:
        """Log raw data collection and quality metrics"""
        if ticker not in self.session_logs:
            return
        
        # Convert DataFrame to serializable format
        data_summary = {
            "shape": raw_data.shape,
            "columns": raw_data.columns.tolist(),
            "date_range": {
                "start": raw_data.index.min().isoformat() if hasattr(raw_data.index, 'min') else None,
                "end": raw_data.index.max().isoformat() if hasattr(raw_data.index, 'max') else None
            },
            "sample_data": raw_data.head().to_dict() if not raw_data.empty else {},
            "missing_values": raw_data.isnull().sum().to_dict(),
            "data_types": raw_data.dtypes.astype(str).to_dict()
        }
        
        self.session_logs[ticker].raw_data = data_summary
        self.session_logs[ticker].data_quality_metrics = quality_metrics
    
    def log_feature_calculation(self, ticker: str, step_name: str, 
                               input_data: Dict[str, Any], method: str,
                               output_data: Dict[str, Any], 
                               processing_time_ms: float, notes: str = "") -> None:
        """Log individual feature calculation step"""
        if ticker not in self.session_logs:
            return
        
        step = CalculationStep(
            step_name=step_name,
            input_data=self._serialize_data(input_data),
            calculation_method=method,
            output_data=self._serialize_data(output_data),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time_ms,
            notes=notes
        )
        
        self.session_logs[ticker].feature_calculations.append(step)
    
    def log_final_features(self, ticker: str, features: Dict[str, float], 
                          statistics: Dict[str, Any]) -> None:
        """Log final feature set and statistics"""
        if ticker not in self.session_logs:
            return
        
        self.session_logs[ticker].final_features = features
        self.session_logs[ticker].feature_statistics = statistics
    
    def log_model_prediction(self, ticker: str, model_name: str, model_version: str,
                           input_features: List[float], feature_names: List[str],
                           raw_prediction: float, confidence_score: float,
                           model_parameters: Dict[str, Any],
                           feature_importance: Dict[str, float],
                           explanation: str, processing_time_ms: float) -> None:
        """Log model prediction with full details"""
        if ticker not in self.session_logs:
            return
        
        model_output = ModelOutput(
            model_name=model_name,
            model_version=model_version,
            input_features=input_features,
            feature_names=feature_names,
            raw_prediction=raw_prediction,
            confidence_score=confidence_score,
            model_parameters=self._serialize_data(model_parameters),
            feature_importance=feature_importance,
            prediction_explanation=explanation,
            processing_time_ms=processing_time_ms
        )
        
        self.session_logs[ticker].model_outputs.append(model_output)
    
    def log_ensemble_calculation(self, ticker: str, model_predictions: Dict[str, float],
                               weights: Dict[str, float], final_prediction: float,
                               method: str, processing_time_ms: float) -> None:
        """Log ensemble model calculation"""
        if ticker not in self.session_logs:
            return
        
        ensemble_step = CalculationStep(
            step_name="Ensemble Prediction",
            input_data={
                "model_predictions": model_predictions,
                "model_weights": weights
            },
            calculation_method=method,
            output_data={
                "final_prediction": final_prediction,
                "weighted_average": sum(pred * weights.get(model, 1.0) 
                                      for model, pred in model_predictions.items()),
                "prediction_variance": np.var(list(model_predictions.values()))
            },
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time_ms,
            notes=f"Combined {len(model_predictions)} model predictions"
        )
        
        self.session_logs[ticker].ensemble_calculation = ensemble_step
    
    def log_llm_analysis(self, ticker: str, stage: str, prompt: str, response: str,
                        parsed_output: Dict[str, Any], confidence: float,
                        reasoning: List[str], processing_time_ms: float,
                        provider: str, model: str) -> None:
        """Log LLM analysis with full prompt and response"""
        if ticker not in self.session_logs:
            return
        
        llm_log = LLMAnalysisLog(
            stage=stage,
            prompt_sent=prompt,
            response_received=response,
            parsed_output=self._serialize_data(parsed_output),
            confidence_level=confidence,
            reasoning=reasoning,
            processing_time_ms=processing_time_ms,
            api_provider=provider,
            model_used=model
        )
        
        self.session_logs[ticker].llm_analyses.append(llm_log)
    
    def log_final_decision(self, ticker: str, recommendation: Dict[str, Any],
                          rationale: str, risk_assessment: Dict[str, Any],
                          total_time_ms: float, system_performance: Dict[str, Any]) -> None:
        """Log final recommendation decision"""
        if ticker not in self.session_logs:
            return
        
        self.session_logs[ticker].final_recommendation = self._serialize_data(recommendation)
        self.session_logs[ticker].decision_rationale = rationale
        self.session_logs[ticker].risk_assessment = self._serialize_data(risk_assessment)
        self.session_logs[ticker].total_processing_time_ms = total_time_ms
        self.session_logs[ticker].system_performance = self._serialize_data(system_performance)
    
    def save_stock_log(self, ticker: str) -> str:
        """Save individual stock transparency log to file"""
        if ticker not in self.session_logs:
            return ""
        
        log_data = asdict(self.session_logs[ticker])
        
        # Save as JSON
        json_filename = f"{self.output_dir}/{self.current_session_id}/{ticker}_transparency_log.json"
        with open(json_filename, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        # Save as human-readable report
        report_filename = f"{self.output_dir}/{self.current_session_id}/{ticker}_analysis_report.md"
        self._generate_markdown_report(ticker, report_filename)
        
        return json_filename
    
    def save_session_summary(self) -> str:
        """Save summary of entire session"""
        summary = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            "total_stocks_analyzed": len(self.session_logs),
            "stocks": list(self.session_logs.keys()),
            "session_statistics": self._calculate_session_statistics()
        }
        
        summary_filename = f"{self.output_dir}/{self.current_session_id}/session_summary.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary_filename
    
    def get_transparency_files(self, ticker: str) -> Dict[str, str]:
        """Get file paths for transparency documents"""
        if ticker not in self.session_logs:
            return {}
        
        return {
            "json_log": f"{self.output_dir}/{self.current_session_id}/{ticker}_transparency_log.json",
            "markdown_report": f"{self.output_dir}/{self.current_session_id}/{ticker}_analysis_report.md",
            "session_summary": f"{self.output_dir}/{self.current_session_id}/session_summary.json"
        }
    
    def _serialize_data(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict()
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif hasattr(data, '__dict__'):
            return {k: self._serialize_data(v) for k, v in data.__dict__.items()}
        else:
            return data
    
    def _calculate_session_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for the entire session"""
        if not self.session_logs:
            return {}
        
        total_time = sum(log.total_processing_time_ms for log in self.session_logs.values())
        model_counts = {}
        feature_counts = {}
        
        for log in self.session_logs.values():
            for model_output in log.model_outputs:
                model_counts[model_output.model_name] = model_counts.get(model_output.model_name, 0) + 1
            
            feature_counts[log.ticker] = len(log.final_features)
        
        return {
            "total_processing_time_ms": total_time,
            "average_processing_time_ms": total_time / len(self.session_logs),
            "models_used": model_counts,
            "average_features_per_stock": np.mean(list(feature_counts.values())),
            "total_llm_calls": sum(len(log.llm_analyses) for log in self.session_logs.values())
        }
    
    def _generate_markdown_report(self, ticker: str, filename: str) -> None:
        """Generate human-readable markdown report"""
        if ticker not in self.session_logs:
            return
        
        log = self.session_logs[ticker]
        
        report = f"""# Transparency Report: {ticker}

## Analysis Overview
- **Ticker**: {ticker}
- **Analysis Time**: {log.analysis_timestamp}
- **Total Processing Time**: {log.total_processing_time_ms:.2f}ms

## User Input
```json
{json.dumps(log.user_input, indent=2)}
```

## Data Collection
- **Data Shape**: {log.raw_data.get('shape', 'N/A')}
- **Date Range**: {log.raw_data.get('date_range', {}).get('start', 'N/A')} to {log.raw_data.get('date_range', {}).get('end', 'N/A')}
- **Data Quality Score**: {log.data_quality_metrics.get('quality_score', 'N/A')}

## Feature Engineering
Total Features Generated: {len(log.final_features)}

### Feature Calculation Steps
"""
        
        for i, step in enumerate(log.feature_calculations, 1):
            report += f"""
#### Step {i}: {step.step_name}
- **Method**: {step.calculation_method}
- **Processing Time**: {step.processing_time_ms:.2f}ms
- **Notes**: {step.notes}
"""
        
        report += f"""
## Model Predictions
Total Models Used: {len(log.model_outputs)}

"""
        
        for model in log.model_outputs:
            report += f"""
### {model.model_name} v{model.model_version}
- **Prediction**: {model.raw_prediction:.4f}
- **Confidence**: {model.confidence_score:.4f}
- **Processing Time**: {model.processing_time_ms:.2f}ms
- **Explanation**: {model.prediction_explanation}

#### Top Feature Importance
"""
            # Show top 5 features
            sorted_features = sorted(model.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                report += f"- **{feature}**: {importance:.4f}\n"
        
        if log.ensemble_calculation:
            report += f"""
## Ensemble Calculation
- **Method**: {log.ensemble_calculation.calculation_method}
- **Final Prediction**: {log.ensemble_calculation.output_data.get('final_prediction', 'N/A')}
- **Processing Time**: {log.ensemble_calculation.processing_time_ms:.2f}ms
"""
        
        report += f"""
## LLM Analysis
Total LLM Calls: {len(log.llm_analyses)}

"""
        
        for i, llm_analysis in enumerate(log.llm_analyses, 1):
            report += f"""
### LLM Analysis {i}: {llm_analysis.stage.title()}
- **Provider**: {llm_analysis.api_provider}
- **Model**: {llm_analysis.model_used}
- **Confidence**: {llm_analysis.confidence_level:.4f}
- **Processing Time**: {llm_analysis.processing_time_ms:.2f}ms

#### Key Reasoning Points
"""
            for reason in llm_analysis.reasoning:
                report += f"- {reason}\n"
        
        report += f"""
## Final Recommendation
```json
{json.dumps(log.final_recommendation, indent=2)}
```

### Decision Rationale
{log.decision_rationale}

### Risk Assessment
```json
{json.dumps(log.risk_assessment, indent=2)}
```

## System Performance
```json
{json.dumps(log.system_performance, indent=2)}
```

---
*Report generated by TAI Transparency System on {datetime.now().isoformat()}*
"""
        
        with open(filename, 'w') as f:
            f.write(report)

# Global transparency logger instance
transparency_logger = TransparencyLogger() 