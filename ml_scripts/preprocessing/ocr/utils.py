"""
OCR Utility Functions

Helper functions for OCR operations and image processing.
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_supported_image_formats() -> List[str]:
    """
    Get list of supported image formats.
    
    Returns:
        List of supported file extensions
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']


def find_images_in_directory(directory_path: str) -> List[str]:
    """
    Find all supported image files in a directory.
    
    Args:
        directory_path: Path to the directory to search
        
    Returns:
        List of image file paths
    """
    supported_formats = get_supported_image_formats()
    image_files = []
    
    try:
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                image_files.append(str(file_path))
        
        logger.info(f"Found {len(image_files)} image files in {directory_path}")
        return image_files
        
    except Exception as e:
        logger.error(f"Error searching for images: {str(e)}")
        return []


def validate_tesseract_installation() -> Dict[str, Any]:
    """
    Validate that Tesseract is properly installed and accessible.
    
    Returns:
        Dictionary containing validation results
    """
    try:
        import pytesseract
        
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        
        # Try to get available languages
        languages = pytesseract.get_languages()
        
        return {
            'installed': True,
            'version': str(version),
            'available_languages': languages,
            'default_language': 'eng' if 'eng' in languages else languages[0] if languages else None
        }
        
    except Exception as e:
        return {
            'installed': False,
            'error': str(e),
            'suggestion': 'Please install Tesseract OCR and ensure it is in your system PATH'
        }


def create_ocr_config(
    page_segmentation_mode: int = 6,
    ocr_engine_mode: int = 3,
    language: str = 'eng',
    whitelist_chars: str = None,
    blacklist_chars: str = None
) -> str:
    """
    Create a custom Tesseract configuration string.
    
    Args:
        page_segmentation_mode: PSM mode (0-13)
        ocr_engine_mode: OEM mode (0-3)
        language: Language code
        whitelist_chars: Characters to whitelist
        blacklist_chars: Characters to blacklist
        
    Returns:
        Tesseract configuration string
    """
    config_parts = [
        f'--oem {ocr_engine_mode}',
        f'--psm {page_segmentation_mode}',
        f'-l {language}'
    ]
    
    if whitelist_chars:
        config_parts.append(f'-c tessedit_char_whitelist={whitelist_chars}')
    
    if blacklist_chars:
        config_parts.append(f'-c tessedit_char_blacklist={blacklist_chars}')
    
    return ' '.join(config_parts)


def get_medical_ocr_config() -> str:
    """
    Get OCR configuration optimized for medical documents.
    
    Returns:
        Tesseract configuration string for medical documents
    """
    return create_ocr_config(
        page_segmentation_mode=6,  # Uniform block of text
        ocr_engine_mode=3,         # Default, based on what is available
        language='eng',
        whitelist_chars='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()-[]{}/"\'@ '
    )


def estimate_processing_time(image_count: int, avg_image_size_mb: float = 1.0) -> Dict[str, float]:
    """
    Estimate processing time for batch OCR operations.
    
    Args:
        image_count: Number of images to process
        avg_image_size_mb: Average image size in MB
        
    Returns:
        Dictionary with time estimates
    """
    # Rough estimates based on typical performance
    base_time_per_image = 2.0  # seconds
    size_factor = max(1.0, avg_image_size_mb * 0.5)
    
    estimated_time_per_image = base_time_per_image * size_factor
    total_estimated_time = estimated_time_per_image * image_count
    
    return {
        'estimated_time_per_image_seconds': round(estimated_time_per_image, 2),
        'total_estimated_time_seconds': round(total_estimated_time, 2),
        'total_estimated_time_minutes': round(total_estimated_time / 60, 2),
        'total_estimated_time_hours': round(total_estimated_time / 3600, 2)
    }