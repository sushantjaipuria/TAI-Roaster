"""
File Handling Utilities

This module provides safe file handling operations for the TAI Roaster API:
- Saving uploaded files to designated directories
- Generating unique filenames to prevent conflicts
- Error handling for file operations

Key features:
- Thread-safe file operations
- Automatic directory creation
- Unique filename generation to prevent overwrites
- Comprehensive error handling
"""

import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileHandler:
    """
    Utility class for handling file operations safely.
    """
    
    def __init__(self):
        """Initialize file handler and ensure directories exist."""
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """
        Create upload directories if they don't exist.
        
        Simple explanation:
        This method creates the folders we need (uploads, temp, processed) 
        if they don't already exist on the filesystem.
        """
        directories = [
            settings.UPLOAD_DIR,
            settings.TEMP_DIR, 
            settings.PROCESSED_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Ensured directory exists: {directory}")
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """
        Generate a unique filename to prevent file overwrites.
        
        Simple explanation:
        Takes a filename like "portfolio.csv" and creates a unique version
        like "20241218_143052_abc123_portfolio.csv" so multiple users
        can upload files with the same name without conflicts.
        
        Args:
            original_filename: The original filename from user upload
            
        Returns:
            Unique filename with timestamp and UUID prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID
        
        # Split filename and extension
        name_parts = original_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            name, extension = name_parts
            unique_filename = f"{timestamp}_{unique_id}_{name}.{extension}"
        else:
            unique_filename = f"{timestamp}_{unique_id}_{original_filename}"
        
        # Clean filename of any problematic characters
        unique_filename = self._sanitize_filename(unique_filename)
        
        logger.debug(f"📝 Generated unique filename: {original_filename} -> {unique_filename}")
        return unique_filename
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Remove or replace problematic characters in filenames.
        
        Simple explanation:
        Makes sure the filename is safe for all operating systems by
        removing special characters that could cause problems.
        """
        # Replace problematic characters with underscores
        problematic_chars = '<>:"/\\|?*'
        for char in problematic_chars:
            filename = filename.replace(char, '_')
        
        # Remove multiple consecutive underscores
        while '__' in filename:
            filename = filename.replace('__', '_')
            
        return filename.strip('_')
    
    async def save_uploaded_file(
        self, 
        file_content: bytes, 
        original_filename: str,
        directory: str = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Save uploaded file content to disk safely.
        
        Simple explanation:
        Takes the raw file data (bytes) and saves it to a file on disk.
        Returns success status, file path, and any error message.
        
        Args:
            file_content: Raw file data as bytes
            original_filename: Original filename from upload
            directory: Target directory (defaults to uploads)
            
        Returns:
            Tuple of (success, file_path, error_message)
        """
        try:
            # Use default upload directory if none specified
            target_dir = directory or settings.UPLOAD_DIR
            
            # Generate unique filename
            unique_filename = self.generate_unique_filename(original_filename)
            
            # Create full file path
            file_path = os.path.join(target_dir, unique_filename)
            
            # Save file content to disk
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Verify file was saved correctly
            if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
                logger.info(f"✅ File saved successfully: {file_path}")
                return True, file_path, None
            else:
                logger.error(f"❌ File verification failed: {file_path}")
                return False, None, "File verification failed after saving"
                
        except PermissionError as e:
            error_msg = f"Permission denied writing to {target_dir}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
            
        except OSError as e:
            error_msg = f"OS error saving file: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error saving file: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return False, None, error_msg


# Create global instance
file_handler = FileHandler()
