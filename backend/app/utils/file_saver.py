"""
File Saver Utility for TAI-Roaster
Saves analysis results to the processed directory for frontend consumption
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AnalysisFileSaver:
    """Handles saving analysis results to the processed directory"""
    
    def __init__(self):
        # Get the processed directory path relative to backend
        self.backend_root = Path(__file__).parent.parent.parent  # Go up to backend root
        self.processed_dir = self.backend_root / "processed"
        
        # Ensure processed directory exists
        self.processed_dir.mkdir(exist_ok=True)
        
        logger.info(f"✅ AnalysisFileSaver initialized - processed directory: {self.processed_dir}")
    
    def save_analysis_result(self, analysis_data: Dict[str, Any], analysis_id: str) -> Tuple[bool, str, str]:
        """
        Save analysis result to processed directory
        
        Args:
            analysis_data: The analysis result in frontend format
            analysis_id: Unique identifier for this analysis
            
        Returns:
            Tuple of (success: bool, file_path: str, error_message: str)
        """
        try:
            # Create filename in the format expected by frontend
            filename = f"analysis_{analysis_id}.json"
            file_path = self.processed_dir / filename
            
            # Add metadata to the analysis data
            analysis_data_with_meta = {
                **analysis_data,
                "analysis_id": analysis_id,
                "generated_at": datetime.now().isoformat(),
                "file_generated_by": "intelligence_module",
                "format_version": "1.0"
            }
            
            # Save to file with proper formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data_with_meta, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Analysis result saved to: {file_path}")
            return True, str(file_path), ""
            
        except Exception as e:
            error_msg = f"Failed to save analysis result: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, "", error_msg
    
    def save_demo_format_analysis(self, analysis_data: Dict[str, Any], analysis_id: str) -> Tuple[bool, str, str]:
        """
        Save analysis in the exact demo format (for compatibility)
        
        This creates a file that matches the exact format the current analysis.py loads
        """
        try:
            # Use the demo naming convention
            filename = f"analysis_demo-{analysis_id}.json"
            file_path = self.processed_dir / filename
            
            # Save without extra metadata (to match demo format exactly)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Demo format analysis saved to: {file_path}")
            return True, str(file_path), ""
            
        except Exception as e:
            error_msg = f"Failed to save demo format analysis: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, "", error_msg
    
    def list_existing_analyses(self) -> list:
        """List all existing analysis files in processed directory"""
        try:
            analysis_files = []
            for file_path in self.processed_dir.glob("analysis_*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    analysis_files.append({
                        "filename": file_path.name,
                        "analysis_id": data.get("analysis_id", file_path.stem),
                        "portfolio_name": data.get("portfolioName", "Unknown"),
                        "analysis_date": data.get("analysisDate", "Unknown"),
                        "generated_at": data.get("generated_at", "Unknown"),
                        "file_size": file_path.stat().st_size
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Could not read analysis file {file_path}: {e}")
                    continue
            
            return analysis_files
            
        except Exception as e:
            logger.error(f"❌ Error listing analyses: {e}")
            return []
    
    def get_analysis_by_id(self, analysis_id: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Retrieve analysis result by ID
        
        Args:
            analysis_id: The analysis ID to retrieve
            
        Returns:
            Tuple of (success: bool, analysis_data: dict, error_message: str)
        """
        try:
            # Try multiple filename patterns
            possible_filenames = [
                f"analysis_{analysis_id}.json",
                f"analysis_demo-{analysis_id}.json",
                f"analysis_{analysis_id}-analysis.json"
            ]
            
            for filename in possible_filenames:
                file_path = self.processed_dir / filename
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    logger.info(f"✅ Retrieved analysis from: {file_path}")
                    return True, data, ""
            
            error_msg = f"Analysis file not found for ID: {analysis_id}"
            logger.warning(f"⚠️ {error_msg}")
            return False, {}, error_msg
            
        except Exception as e:
            error_msg = f"Failed to retrieve analysis {analysis_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, {}, error_msg
    
    def delete_analysis(self, analysis_id: str) -> Tuple[bool, str]:
        """
        Delete analysis file by ID
        
        Args:
            analysis_id: The analysis ID to delete
            
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            # Try multiple filename patterns
            possible_filenames = [
                f"analysis_{analysis_id}.json",
                f"analysis_demo-{analysis_id}.json",
                f"analysis_{analysis_id}-analysis.json"
            ]
            
            deleted_files = []
            for filename in possible_filenames:
                file_path = self.processed_dir / filename
                if file_path.exists():
                    file_path.unlink()
                    deleted_files.append(filename)
            
            if deleted_files:
                logger.info(f"✅ Deleted analysis files: {deleted_files}")
                return True, ""
            else:
                error_msg = f"No analysis files found for ID: {analysis_id}"
                logger.warning(f"⚠️ {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Failed to delete analysis {analysis_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    def cleanup_old_analyses(self, keep_count: int = 10) -> Tuple[int, str]:
        """
        Clean up old analysis files, keeping only the most recent ones
        
        Args:
            keep_count: Number of most recent analyses to keep
            
        Returns:
            Tuple of (deleted_count: int, error_message: str)
        """
        try:
            # Get all analysis files sorted by modification time (newest first)
            analysis_files = list(self.processed_dir.glob("analysis_*.json"))
            analysis_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Delete files beyond the keep_count
            deleted_count = 0
            for file_path in analysis_files[keep_count:]:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted old analysis: {file_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"✅ Cleanup completed - deleted {deleted_count} old analysis files")
            
            return deleted_count, ""
            
        except Exception as e:
            error_msg = f"Failed to cleanup old analyses: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return 0, error_msg
    
    def get_processed_directory_info(self) -> Dict[str, Any]:
        """Get information about the processed directory"""
        try:
            analysis_files = list(self.processed_dir.glob("analysis_*.json"))
            total_size = sum(f.stat().st_size for f in analysis_files)
            
            return {
                "directory_path": str(self.processed_dir),
                "exists": self.processed_dir.exists(),
                "is_writable": os.access(self.processed_dir, os.W_OK),
                "analysis_count": len(analysis_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": [f.name for f in analysis_files]
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting directory info: {e}")
            return {"error": str(e)}

# Create a singleton instance
analysis_file_saver = AnalysisFileSaver()
