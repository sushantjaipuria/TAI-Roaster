"""
Job Scheduler - Automated Daily Evaluation Runner
Handles automated scheduling and execution of daily evaluation tasks.

Features from PRD:
- Daily evaluation at 00:30 IST
- Automated prediction generation
- Error handling and retry logic
- Email notifications (optional)
- Comprehensive logging
- Configuration management
"""

import schedule
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio
import threading
import traceback
from dataclasses import dataclass
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import os

# Import evaluation components
from .historical_evaluator import HistoricalEvaluator, run_daily_evaluation
from .results_logger import ResultsLogger

# Import TAI system components
import sys
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig:
    """Configuration for the evaluation scheduler."""
    evaluation_time: str = "00:30"  # IST time for daily evaluation
    timezone: str = "Asia/Kolkata"
    retry_attempts: int = 3
    retry_delay_minutes: int = 15
    cutoff_date: str = "2024-06-30"
    investment_amount: float = 100000
    output_base_dir: str = "logs/evaluation"
    enable_email_notifications: bool = False
    email_config: Optional[Dict[str, str]] = None
    log_level: str = "INFO"
    generate_fresh_predictions: bool = True
    multi_horizon_evaluation: bool = False
    
class EvaluationJobScheduler:
    """
    Automated job scheduler for daily model evaluations.
    
    Features:
    - Automated daily evaluation runs
    - Error handling and retry logic
    - Email notifications for results/failures
    - Comprehensive logging and monitoring
    - Configuration management
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Initialize the evaluation job scheduler.
        
        Args:
            config (Optional[SchedulerConfig]): Scheduler configuration
        """
        self.config = config or SchedulerConfig()
        self.is_running = False
        self.job_history: List[Dict[str, Any]] = []
        self.current_job_id = None
        self._setup_logging()
        
        logger.info(f"EvaluationJobScheduler initialized:")
        logger.info(f"  Evaluation Time: {self.config.evaluation_time} {self.config.timezone}")
        logger.info(f"  Cutoff Date: {self.config.cutoff_date}")
        logger.info(f"  Investment Amount: ₹{self.config.investment_amount:,.2f}")
        logger.info(f"  Email Notifications: {self.config.enable_email_notifications}")
    
    def _setup_logging(self):
        """Setup comprehensive logging for the scheduler."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create logs directory
        log_dir = Path("logs/scheduler")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for daily logs
        file_handler = logging.FileHandler(
            log_dir / f"scheduler_{datetime.now().strftime('%Y_%m')}.log"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Add handlers if not already present
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            root_logger.addHandler(file_handler)
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            root_logger.addHandler(console_handler)
    
    def schedule_daily_evaluation(self):
        """Schedule daily evaluation job."""
        logger.info(f"🕐 Scheduling daily evaluation at {self.config.evaluation_time}")
        
        schedule.every().day.at(self.config.evaluation_time).do(
            self._run_daily_evaluation_job
        )
        
        logger.info("✅ Daily evaluation scheduled successfully")
    
    def schedule_weekly_multi_horizon(self, weekday: str = "sunday", time: str = "02:00"):
        """Schedule weekly multi-horizon evaluation."""
        if self.config.multi_horizon_evaluation:
            logger.info(f"🕐 Scheduling weekly multi-horizon evaluation on {weekday} at {time}")
            
            getattr(schedule.every(), weekday.lower()).at(time).do(
                self._run_multi_horizon_evaluation_job
            )
            
            logger.info("✅ Weekly multi-horizon evaluation scheduled")
    
    def _run_daily_evaluation_job(self) -> Dict[str, Any]:
        """Execute daily evaluation job with error handling and retry logic."""
        job_id = f"daily_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_job_id = job_id
        
        job_result = {
            "job_id": job_id,
            "job_type": "daily_evaluation",
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "attempts": 0,
            "errors": [],
            "result": None
        }
        
        logger.info(f"🚀 Starting daily evaluation job: {job_id}")
        
        # Attempt evaluation with retry logic
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                job_result["attempts"] = attempt
                logger.info(f"📊 Evaluation attempt {attempt}/{self.config.retry_attempts}")
                
                # Generate or load predictions
                predictions_source = self._get_predictions_source()
                
                # Run evaluation
                evaluator = HistoricalEvaluator(
                    cutoff_date=self.config.cutoff_date,
                    output_dir=self.config.output_base_dir
                )
                
                evaluation_result = evaluator.run_comprehensive_evaluation(
                    predictions_source=predictions_source,
                    model_metadata={
                        "scheduled_job_id": job_id,
                        "evaluation_type": "daily_automated",
                        "scheduler_version": "1.0"
                    },
                    investment_amount=self.config.investment_amount
                )
                
                # Job completed successfully
                job_result.update({
                    "status": "completed",
                    "end_time": datetime.now().isoformat(),
                    "result": evaluation_result,
                    "success": True
                })
                
                logger.info(f"✅ Daily evaluation job completed successfully: {job_id}")
                
                # Send success notification
                self._send_notification(job_result, success=True)
                
                break  # Exit retry loop on success
                
            except Exception as e:
                error_msg = f"Attempt {attempt} failed: {str(e)}"
                job_result["errors"].append({
                    "attempt": attempt,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat(),
                    "traceback": traceback.format_exc()
                })
                
                logger.error(f"❌ {error_msg}")
                
                if attempt < self.config.retry_attempts:
                    logger.info(f"⏳ Retrying in {self.config.retry_delay_minutes} minutes...")
                    time.sleep(self.config.retry_delay_minutes * 60)
                else:
                    # All attempts failed
                    job_result.update({
                        "status": "failed",
                        "end_time": datetime.now().isoformat(),
                        "success": False
                    })
                    
                    logger.error(f"❌ Daily evaluation job failed after {self.config.retry_attempts} attempts: {job_id}")
                    
                    # Send failure notification
                    self._send_notification(job_result, success=False)
        
        # Store job history
        self.job_history.append(job_result)
        self._save_job_history()
        
        self.current_job_id = None
        return job_result
    
    def _run_multi_horizon_evaluation_job(self) -> Dict[str, Any]:
        """Execute multi-horizon evaluation job."""
        job_id = f"multi_horizon_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_job_id = job_id
        
        job_result = {
            "job_id": job_id,
            "job_type": "multi_horizon_evaluation",
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "attempts": 1,
            "errors": [],
            "result": None
        }
        
        logger.info(f"🚀 Starting multi-horizon evaluation job: {job_id}")
        
        try:
            # Generate or load predictions
            predictions_source = self._get_predictions_source()
            
            # Run multi-horizon evaluation
            evaluator = HistoricalEvaluator(
                cutoff_date=self.config.cutoff_date,
                output_dir=self.config.output_base_dir
            )
            
            evaluation_result = evaluator.evaluate_multiple_horizons(
                predictions_source=predictions_source,
                investment_amount=self.config.investment_amount
            )
            
            # Job completed successfully
            job_result.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "result": evaluation_result,
                "success": True
            })
            
            logger.info(f"✅ Multi-horizon evaluation job completed: {job_id}")
            
            # Send success notification
            self._send_notification(job_result, success=True)
            
        except Exception as e:
            error_msg = f"Multi-horizon evaluation failed: {str(e)}"
            job_result.update({
                "status": "failed",
                "end_time": datetime.now().isoformat(),
                "success": False,
                "errors": [{
                    "attempt": 1,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat(),
                    "traceback": traceback.format_exc()
                }]
            })
            
            logger.error(f"❌ {error_msg}")
            
            # Send failure notification
            self._send_notification(job_result, success=False)
        
        # Store job history
        self.job_history.append(job_result)
        self._save_job_history()
        
        self.current_job_id = None
        return job_result
    
    def _get_predictions_source(self) -> Any:
        """Get predictions source (generate fresh or load existing)."""
        if self.config.generate_fresh_predictions:
            logger.info("🔄 Generating fresh predictions from TAI system...")
            
            try:
                # Import TAI pipeline
                from intelligence.pipeline import run_full_pipeline
                from intelligence.schemas import UserInput
                
                # Create user input for prediction generation
                user_input = UserInput(
                    investment_amount=self.config.investment_amount,
                    risk_tolerance="medium",
                    market_cap="largecap"
                )
                
                # Generate predictions
                pipeline_results = run_full_pipeline(user_input)
                
                # Save predictions for future reference
                predictions_file = Path(self.config.output_base_dir) / "daily_predictions" / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
                predictions_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(predictions_file, 'w') as f:
                    json.dump(pipeline_results, f, indent=2, default=str)
                
                logger.info(f"✅ Fresh predictions generated and saved to: {predictions_file}")
                return pipeline_results
                
            except Exception as e:
                logger.error(f"❌ Failed to generate fresh predictions: {e}")
                logger.info("🔄 Falling back to existing predictions...")
                return self._load_latest_predictions()
        else:
            return self._load_latest_predictions()
    
    def _load_latest_predictions(self) -> str:
        """Load the latest available predictions file."""
        predictions_dir = Path(self.config.output_base_dir) / "daily_predictions"
        
        if not predictions_dir.exists():
            raise FileNotFoundError("No predictions directory found")
        
        # Find latest predictions file
        prediction_files = list(predictions_dir.glob("predictions_*.json"))
        
        if not prediction_files:
            raise FileNotFoundError("No prediction files found")
        
        latest_file = max(prediction_files, key=lambda f: f.stat().st_mtime)
        
        logger.info(f"📂 Loading latest predictions from: {latest_file}")
        return str(latest_file)
    
    def _send_notification(self, job_result: Dict[str, Any], success: bool):
        """Send email notification about job completion."""
        if not self.config.enable_email_notifications or not self.config.email_config:
            return
        
        try:
            subject = f"TAI Evaluation {'Success' if success else 'Failed'} - {job_result['job_id']}"
            
            # Create email content
            body = self._create_email_body(job_result, success)
            
            # Send email
            self._send_email(subject, body)
            
            logger.info(f"📧 Notification sent: {subject}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send notification: {e}")
    
    def _create_email_body(self, job_result: Dict[str, Any], success: bool) -> str:
        """Create email body for notification."""
        if success:
            result = job_result.get("result", {})
            portfolio_perf = result.get("portfolio_performance", {})
            
            body = f"""
TAI Trading System - Evaluation Report

Job ID: {job_result['job_id']}
Job Type: {job_result['job_type']}
Status: ✅ SUCCESS
Duration: {job_result.get('start_time', '')} to {job_result.get('end_time', '')}

Performance Summary:
- Total Return: {portfolio_perf.get('total_return', 0):.2f}%
- Sharpe Ratio: {portfolio_perf.get('sharpe_ratio_daily', 0):.2f}
- Max Drawdown: {portfolio_perf.get('max_drawdown', {}).get('max_drawdown', 0):.2f}%

Data Quality:
- Aligned Pairs: {result.get('data_quality', {}).get('aligned_pairs', 0)}
- Data Coverage: {result.get('data_quality', {}).get('data_coverage', 0):.1%}

Reports Generated: {len(result.get('generated_reports', {}))} files

This is an automated notification from the TAI Evaluation Engine.
            """
        else:
            errors = job_result.get("errors", [])
            last_error = errors[-1] if errors else {}
            
            body = f"""
TAI Trading System - Evaluation FAILED

Job ID: {job_result['job_id']}
Job Type: {job_result['job_type']}
Status: ❌ FAILED
Attempts: {job_result.get('attempts', 0)}/{self.config.retry_attempts}

Last Error:
{last_error.get('error', 'Unknown error')}

Error Time: {last_error.get('timestamp', 'Unknown')}

Please check the logs for more details.

This is an automated notification from the TAI Evaluation Engine.
            """
        
        return body
    
    def _send_email(self, subject: str, body: str):
        """Send email notification."""
        if not self.config.email_config:
            return
        
        msg = MimeMultipart()
        msg['From'] = self.config.email_config['from_email']
        msg['To'] = self.config.email_config['to_email']
        msg['Subject'] = subject
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(
            self.config.email_config['smtp_server'],
            self.config.email_config['smtp_port']
        )
        
        if self.config.email_config.get('use_tls', True):
            server.starttls()
        
        server.login(
            self.config.email_config['username'],
            self.config.email_config['password']
        )
        
        server.send_message(msg)
        server.quit()
    
    def _save_job_history(self):
        """Save job history to file."""
        history_file = Path(self.config.output_base_dir) / "job_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Keep only last 100 jobs
        recent_history = self.job_history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(recent_history, f, indent=2, default=str)
    
    def start_scheduler(self):
        """Start the scheduler in a separate thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        # Schedule jobs
        self.schedule_daily_evaluation()
        self.schedule_weekly_multi_horizon()
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("✅ Evaluation scheduler started successfully")
        logger.info(f"📅 Next daily evaluation: {schedule.next_run()}")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        logger.info("🔄 Scheduler loop started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.is_running = False
        schedule.clear()
        logger.info("⏹️ Scheduler stopped")
    
    def run_immediate_evaluation(self, evaluation_type: str = "daily") -> Dict[str, Any]:
        """Run immediate evaluation (for testing or manual trigger)."""
        logger.info(f"🚀 Running immediate {evaluation_type} evaluation...")
        
        if evaluation_type == "daily":
            return self._run_daily_evaluation_job()
        elif evaluation_type == "multi_horizon":
            return self._run_multi_horizon_evaluation_job()
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get current job status and history."""
        return {
            "is_running": self.is_running,
            "current_job_id": self.current_job_id,
            "total_jobs": len(self.job_history),
            "successful_jobs": sum(1 for job in self.job_history if job.get("success", False)),
            "failed_jobs": sum(1 for job in self.job_history if not job.get("success", True)),
            "last_job": self.job_history[-1] if self.job_history else None,
            "next_scheduled_run": str(schedule.next_run()) if schedule.jobs else None
        }
    
    def get_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent job history."""
        return self.job_history[-limit:]

# Configuration helper functions
def create_scheduler_config_from_env() -> SchedulerConfig:
    """Create scheduler configuration from environment variables."""
    return SchedulerConfig(
        evaluation_time=os.getenv("TAI_EVAL_TIME", "00:30"),
        cutoff_date=os.getenv("TAI_CUTOFF_DATE", "2024-06-30"),
        investment_amount=float(os.getenv("TAI_INVESTMENT_AMOUNT", "100000")),
        enable_email_notifications=os.getenv("TAI_EMAIL_NOTIFICATIONS", "false").lower() == "true",
        email_config={
            "smtp_server": os.getenv("SMTP_SERVER"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from_email": os.getenv("FROM_EMAIL"),
            "to_email": os.getenv("TO_EMAIL"),
            "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        } if os.getenv("SMTP_SERVER") else None,
        generate_fresh_predictions=os.getenv("TAI_FRESH_PREDICTIONS", "true").lower() == "true",
        multi_horizon_evaluation=os.getenv("TAI_MULTI_HORIZON", "false").lower() == "true"
    )

def save_scheduler_config(config: SchedulerConfig, config_file: str = "scheduler_config.json"):
    """Save scheduler configuration to file."""
    config_data = {
        "evaluation_time": config.evaluation_time,
        "timezone": config.timezone,
        "retry_attempts": config.retry_attempts,
        "retry_delay_minutes": config.retry_delay_minutes,
        "cutoff_date": config.cutoff_date,
        "investment_amount": config.investment_amount,
        "output_base_dir": config.output_base_dir,
        "enable_email_notifications": config.enable_email_notifications,
        "email_config": config.email_config,
        "log_level": config.log_level,
        "generate_fresh_predictions": config.generate_fresh_predictions,
        "multi_horizon_evaluation": config.multi_horizon_evaluation
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

def load_scheduler_config(config_file: str = "scheduler_config.json") -> SchedulerConfig:
    """Load scheduler configuration from file."""
    if not Path(config_file).exists():
        return SchedulerConfig()
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    return SchedulerConfig(**config_data)

# Main entry point for command-line usage
def main():
    """Main entry point for scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TAI Evaluation Job Scheduler")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--run-immediate", choices=["daily", "multi_horizon"], help="Run immediate evaluation")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--stop", action="store_true", help="Stop running scheduler")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    if args.config:
        config = load_scheduler_config(args.config)
    else:
        config = create_scheduler_config_from_env()
    
    scheduler = EvaluationJobScheduler(config)
    
    if args.run_immediate:
        # Run immediate evaluation
        result = scheduler.run_immediate_evaluation(args.run_immediate)
        print(f"Evaluation completed: {result['status']}")
        
    elif args.status:
        # Show status
        status = scheduler.get_job_status()
        print(json.dumps(status, indent=2, default=str))
        
    elif args.stop:
        # Stop scheduler (placeholder - would need process management)
        print("Scheduler stop command issued")
        
    else:
        # Start scheduler
        scheduler.start_scheduler()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
            scheduler.stop_scheduler()

if __name__ == "__main__":
    main() 
Job Scheduler - Automated Daily Evaluation Runner
Handles automated scheduling and execution of daily evaluation tasks.

Features from PRD:
- Daily evaluation at 00:30 IST
- Automated prediction generation
- Error handling and retry logic
- Email notifications (optional)
- Comprehensive logging
- Configuration management
"""

import schedule
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio
import threading
import traceback
from dataclasses import dataclass
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import os

# Import evaluation components
from .historical_evaluator import HistoricalEvaluator, run_daily_evaluation
from .results_logger import ResultsLogger

# Import TAI system components
import sys
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig:
    """Configuration for the evaluation scheduler."""
    evaluation_time: str = "00:30"  # IST time for daily evaluation
    timezone: str = "Asia/Kolkata"
    retry_attempts: int = 3
    retry_delay_minutes: int = 15
    cutoff_date: str = "2024-06-30"
    investment_amount: float = 100000
    output_base_dir: str = "logs/evaluation"
    enable_email_notifications: bool = False
    email_config: Optional[Dict[str, str]] = None
    log_level: str = "INFO"
    generate_fresh_predictions: bool = True
    multi_horizon_evaluation: bool = False
    
class EvaluationJobScheduler:
    """
    Automated job scheduler for daily model evaluations.
    
    Features:
    - Automated daily evaluation runs
    - Error handling and retry logic
    - Email notifications for results/failures
    - Comprehensive logging and monitoring
    - Configuration management
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Initialize the evaluation job scheduler.
        
        Args:
            config (Optional[SchedulerConfig]): Scheduler configuration
        """
        self.config = config or SchedulerConfig()
        self.is_running = False
        self.job_history: List[Dict[str, Any]] = []
        self.current_job_id = None
        self._setup_logging()
        
        logger.info(f"EvaluationJobScheduler initialized:")
        logger.info(f"  Evaluation Time: {self.config.evaluation_time} {self.config.timezone}")
        logger.info(f"  Cutoff Date: {self.config.cutoff_date}")
        logger.info(f"  Investment Amount: ₹{self.config.investment_amount:,.2f}")
        logger.info(f"  Email Notifications: {self.config.enable_email_notifications}")
    
    def _setup_logging(self):
        """Setup comprehensive logging for the scheduler."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create logs directory
        log_dir = Path("logs/scheduler")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for daily logs
        file_handler = logging.FileHandler(
            log_dir / f"scheduler_{datetime.now().strftime('%Y_%m')}.log"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Add handlers if not already present
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            root_logger.addHandler(file_handler)
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            root_logger.addHandler(console_handler)
    
    def schedule_daily_evaluation(self):
        """Schedule daily evaluation job."""
        logger.info(f"🕐 Scheduling daily evaluation at {self.config.evaluation_time}")
        
        schedule.every().day.at(self.config.evaluation_time).do(
            self._run_daily_evaluation_job
        )
        
        logger.info("✅ Daily evaluation scheduled successfully")
    
    def schedule_weekly_multi_horizon(self, weekday: str = "sunday", time: str = "02:00"):
        """Schedule weekly multi-horizon evaluation."""
        if self.config.multi_horizon_evaluation:
            logger.info(f"🕐 Scheduling weekly multi-horizon evaluation on {weekday} at {time}")
            
            getattr(schedule.every(), weekday.lower()).at(time).do(
                self._run_multi_horizon_evaluation_job
            )
            
            logger.info("✅ Weekly multi-horizon evaluation scheduled")
    
    def _run_daily_evaluation_job(self) -> Dict[str, Any]:
        """Execute daily evaluation job with error handling and retry logic."""
        job_id = f"daily_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_job_id = job_id
        
        job_result = {
            "job_id": job_id,
            "job_type": "daily_evaluation",
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "attempts": 0,
            "errors": [],
            "result": None
        }
        
        logger.info(f"🚀 Starting daily evaluation job: {job_id}")
        
        # Attempt evaluation with retry logic
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                job_result["attempts"] = attempt
                logger.info(f"📊 Evaluation attempt {attempt}/{self.config.retry_attempts}")
                
                # Generate or load predictions
                predictions_source = self._get_predictions_source()
                
                # Run evaluation
                evaluator = HistoricalEvaluator(
                    cutoff_date=self.config.cutoff_date,
                    output_dir=self.config.output_base_dir
                )
                
                evaluation_result = evaluator.run_comprehensive_evaluation(
                    predictions_source=predictions_source,
                    model_metadata={
                        "scheduled_job_id": job_id,
                        "evaluation_type": "daily_automated",
                        "scheduler_version": "1.0"
                    },
                    investment_amount=self.config.investment_amount
                )
                
                # Job completed successfully
                job_result.update({
                    "status": "completed",
                    "end_time": datetime.now().isoformat(),
                    "result": evaluation_result,
                    "success": True
                })
                
                logger.info(f"✅ Daily evaluation job completed successfully: {job_id}")
                
                # Send success notification
                self._send_notification(job_result, success=True)
                
                break  # Exit retry loop on success
                
            except Exception as e:
                error_msg = f"Attempt {attempt} failed: {str(e)}"
                job_result["errors"].append({
                    "attempt": attempt,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat(),
                    "traceback": traceback.format_exc()
                })
                
                logger.error(f"❌ {error_msg}")
                
                if attempt < self.config.retry_attempts:
                    logger.info(f"⏳ Retrying in {self.config.retry_delay_minutes} minutes...")
                    time.sleep(self.config.retry_delay_minutes * 60)
                else:
                    # All attempts failed
                    job_result.update({
                        "status": "failed",
                        "end_time": datetime.now().isoformat(),
                        "success": False
                    })
                    
                    logger.error(f"❌ Daily evaluation job failed after {self.config.retry_attempts} attempts: {job_id}")
                    
                    # Send failure notification
                    self._send_notification(job_result, success=False)
        
        # Store job history
        self.job_history.append(job_result)
        self._save_job_history()
        
        self.current_job_id = None
        return job_result
    
    def _run_multi_horizon_evaluation_job(self) -> Dict[str, Any]:
        """Execute multi-horizon evaluation job."""
        job_id = f"multi_horizon_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_job_id = job_id
        
        job_result = {
            "job_id": job_id,
            "job_type": "multi_horizon_evaluation",
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "attempts": 1,
            "errors": [],
            "result": None
        }
        
        logger.info(f"🚀 Starting multi-horizon evaluation job: {job_id}")
        
        try:
            # Generate or load predictions
            predictions_source = self._get_predictions_source()
            
            # Run multi-horizon evaluation
            evaluator = HistoricalEvaluator(
                cutoff_date=self.config.cutoff_date,
                output_dir=self.config.output_base_dir
            )
            
            evaluation_result = evaluator.evaluate_multiple_horizons(
                predictions_source=predictions_source,
                investment_amount=self.config.investment_amount
            )
            
            # Job completed successfully
            job_result.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "result": evaluation_result,
                "success": True
            })
            
            logger.info(f"✅ Multi-horizon evaluation job completed: {job_id}")
            
            # Send success notification
            self._send_notification(job_result, success=True)
            
        except Exception as e:
            error_msg = f"Multi-horizon evaluation failed: {str(e)}"
            job_result.update({
                "status": "failed",
                "end_time": datetime.now().isoformat(),
                "success": False,
                "errors": [{
                    "attempt": 1,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat(),
                    "traceback": traceback.format_exc()
                }]
            })
            
            logger.error(f"❌ {error_msg}")
            
            # Send failure notification
            self._send_notification(job_result, success=False)
        
        # Store job history
        self.job_history.append(job_result)
        self._save_job_history()
        
        self.current_job_id = None
        return job_result
    
    def _get_predictions_source(self) -> Any:
        """Get predictions source (generate fresh or load existing)."""
        if self.config.generate_fresh_predictions:
            logger.info("🔄 Generating fresh predictions from TAI system...")
            
            try:
                # Import TAI pipeline
                from intelligence.pipeline import run_full_pipeline
                from intelligence.schemas import UserInput
                
                # Create user input for prediction generation
                user_input = UserInput(
                    investment_amount=self.config.investment_amount,
                    risk_tolerance="medium",
                    market_cap="largecap"
                )
                
                # Generate predictions
                pipeline_results = run_full_pipeline(user_input)
                
                # Save predictions for future reference
                predictions_file = Path(self.config.output_base_dir) / "daily_predictions" / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
                predictions_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(predictions_file, 'w') as f:
                    json.dump(pipeline_results, f, indent=2, default=str)
                
                logger.info(f"✅ Fresh predictions generated and saved to: {predictions_file}")
                return pipeline_results
                
            except Exception as e:
                logger.error(f"❌ Failed to generate fresh predictions: {e}")
                logger.info("🔄 Falling back to existing predictions...")
                return self._load_latest_predictions()
        else:
            return self._load_latest_predictions()
    
    def _load_latest_predictions(self) -> str:
        """Load the latest available predictions file."""
        predictions_dir = Path(self.config.output_base_dir) / "daily_predictions"
        
        if not predictions_dir.exists():
            raise FileNotFoundError("No predictions directory found")
        
        # Find latest predictions file
        prediction_files = list(predictions_dir.glob("predictions_*.json"))
        
        if not prediction_files:
            raise FileNotFoundError("No prediction files found")
        
        latest_file = max(prediction_files, key=lambda f: f.stat().st_mtime)
        
        logger.info(f"📂 Loading latest predictions from: {latest_file}")
        return str(latest_file)
    
    def _send_notification(self, job_result: Dict[str, Any], success: bool):
        """Send email notification about job completion."""
        if not self.config.enable_email_notifications or not self.config.email_config:
            return
        
        try:
            subject = f"TAI Evaluation {'Success' if success else 'Failed'} - {job_result['job_id']}"
            
            # Create email content
            body = self._create_email_body(job_result, success)
            
            # Send email
            self._send_email(subject, body)
            
            logger.info(f"📧 Notification sent: {subject}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send notification: {e}")
    
    def _create_email_body(self, job_result: Dict[str, Any], success: bool) -> str:
        """Create email body for notification."""
        if success:
            result = job_result.get("result", {})
            portfolio_perf = result.get("portfolio_performance", {})
            
            body = f"""
TAI Trading System - Evaluation Report

Job ID: {job_result['job_id']}
Job Type: {job_result['job_type']}
Status: ✅ SUCCESS
Duration: {job_result.get('start_time', '')} to {job_result.get('end_time', '')}

Performance Summary:
- Total Return: {portfolio_perf.get('total_return', 0):.2f}%
- Sharpe Ratio: {portfolio_perf.get('sharpe_ratio_daily', 0):.2f}
- Max Drawdown: {portfolio_perf.get('max_drawdown', {}).get('max_drawdown', 0):.2f}%

Data Quality:
- Aligned Pairs: {result.get('data_quality', {}).get('aligned_pairs', 0)}
- Data Coverage: {result.get('data_quality', {}).get('data_coverage', 0):.1%}

Reports Generated: {len(result.get('generated_reports', {}))} files

This is an automated notification from the TAI Evaluation Engine.
            """
        else:
            errors = job_result.get("errors", [])
            last_error = errors[-1] if errors else {}
            
            body = f"""
TAI Trading System - Evaluation FAILED

Job ID: {job_result['job_id']}
Job Type: {job_result['job_type']}
Status: ❌ FAILED
Attempts: {job_result.get('attempts', 0)}/{self.config.retry_attempts}

Last Error:
{last_error.get('error', 'Unknown error')}

Error Time: {last_error.get('timestamp', 'Unknown')}

Please check the logs for more details.

This is an automated notification from the TAI Evaluation Engine.
            """
        
        return body
    
    def _send_email(self, subject: str, body: str):
        """Send email notification."""
        if not self.config.email_config:
            return
        
        msg = MimeMultipart()
        msg['From'] = self.config.email_config['from_email']
        msg['To'] = self.config.email_config['to_email']
        msg['Subject'] = subject
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(
            self.config.email_config['smtp_server'],
            self.config.email_config['smtp_port']
        )
        
        if self.config.email_config.get('use_tls', True):
            server.starttls()
        
        server.login(
            self.config.email_config['username'],
            self.config.email_config['password']
        )
        
        server.send_message(msg)
        server.quit()
    
    def _save_job_history(self):
        """Save job history to file."""
        history_file = Path(self.config.output_base_dir) / "job_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Keep only last 100 jobs
        recent_history = self.job_history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(recent_history, f, indent=2, default=str)
    
    def start_scheduler(self):
        """Start the scheduler in a separate thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        # Schedule jobs
        self.schedule_daily_evaluation()
        self.schedule_weekly_multi_horizon()
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("✅ Evaluation scheduler started successfully")
        logger.info(f"📅 Next daily evaluation: {schedule.next_run()}")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        logger.info("🔄 Scheduler loop started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.is_running = False
        schedule.clear()
        logger.info("⏹️ Scheduler stopped")
    
    def run_immediate_evaluation(self, evaluation_type: str = "daily") -> Dict[str, Any]:
        """Run immediate evaluation (for testing or manual trigger)."""
        logger.info(f"🚀 Running immediate {evaluation_type} evaluation...")
        
        if evaluation_type == "daily":
            return self._run_daily_evaluation_job()
        elif evaluation_type == "multi_horizon":
            return self._run_multi_horizon_evaluation_job()
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get current job status and history."""
        return {
            "is_running": self.is_running,
            "current_job_id": self.current_job_id,
            "total_jobs": len(self.job_history),
            "successful_jobs": sum(1 for job in self.job_history if job.get("success", False)),
            "failed_jobs": sum(1 for job in self.job_history if not job.get("success", True)),
            "last_job": self.job_history[-1] if self.job_history else None,
            "next_scheduled_run": str(schedule.next_run()) if schedule.jobs else None
        }
    
    def get_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent job history."""
        return self.job_history[-limit:]

# Configuration helper functions
def create_scheduler_config_from_env() -> SchedulerConfig:
    """Create scheduler configuration from environment variables."""
    return SchedulerConfig(
        evaluation_time=os.getenv("TAI_EVAL_TIME", "00:30"),
        cutoff_date=os.getenv("TAI_CUTOFF_DATE", "2024-06-30"),
        investment_amount=float(os.getenv("TAI_INVESTMENT_AMOUNT", "100000")),
        enable_email_notifications=os.getenv("TAI_EMAIL_NOTIFICATIONS", "false").lower() == "true",
        email_config={
            "smtp_server": os.getenv("SMTP_SERVER"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from_email": os.getenv("FROM_EMAIL"),
            "to_email": os.getenv("TO_EMAIL"),
            "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        } if os.getenv("SMTP_SERVER") else None,
        generate_fresh_predictions=os.getenv("TAI_FRESH_PREDICTIONS", "true").lower() == "true",
        multi_horizon_evaluation=os.getenv("TAI_MULTI_HORIZON", "false").lower() == "true"
    )

def save_scheduler_config(config: SchedulerConfig, config_file: str = "scheduler_config.json"):
    """Save scheduler configuration to file."""
    config_data = {
        "evaluation_time": config.evaluation_time,
        "timezone": config.timezone,
        "retry_attempts": config.retry_attempts,
        "retry_delay_minutes": config.retry_delay_minutes,
        "cutoff_date": config.cutoff_date,
        "investment_amount": config.investment_amount,
        "output_base_dir": config.output_base_dir,
        "enable_email_notifications": config.enable_email_notifications,
        "email_config": config.email_config,
        "log_level": config.log_level,
        "generate_fresh_predictions": config.generate_fresh_predictions,
        "multi_horizon_evaluation": config.multi_horizon_evaluation
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

def load_scheduler_config(config_file: str = "scheduler_config.json") -> SchedulerConfig:
    """Load scheduler configuration from file."""
    if not Path(config_file).exists():
        return SchedulerConfig()
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    return SchedulerConfig(**config_data)

# Main entry point for command-line usage
def main():
    """Main entry point for scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TAI Evaluation Job Scheduler")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--run-immediate", choices=["daily", "multi_horizon"], help="Run immediate evaluation")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--stop", action="store_true", help="Stop running scheduler")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    if args.config:
        config = load_scheduler_config(args.config)
    else:
        config = create_scheduler_config_from_env()
    
    scheduler = EvaluationJobScheduler(config)
    
    if args.run_immediate:
        # Run immediate evaluation
        result = scheduler.run_immediate_evaluation(args.run_immediate)
        print(f"Evaluation completed: {result['status']}")
        
    elif args.status:
        # Show status
        status = scheduler.get_job_status()
        print(json.dumps(status, indent=2, default=str))
        
    elif args.stop:
        # Stop scheduler (placeholder - would need process management)
        print("Scheduler stop command issued")
        
    else:
        # Start scheduler
        scheduler.start_scheduler()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
            scheduler.stop_scheduler()

if __name__ == "__main__":
    main() 