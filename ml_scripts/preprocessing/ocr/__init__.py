"""
OCR Module

This module handles Optical Character Recognition using Tesseract.
Handles initial image processing and text extraction from medical images.

Main Components:
- TesseractOCR: Main OCR class with preprocessing capabilities
- Quick functions: Simplified OCR operations
- Utility functions: Configuration and validation helpers
"""

# Import main OCR class
from .tesseract_ocr import TesseractOCR, quick_ocr, medical_document_ocr

# Import utility functions
from .utils import (
    configure_tesseract,
    validate_tesseract_installation,
    find_images_in_directory,
    get_supported_image_formats,
    create_ocr_config,
    get_medical_ocr_config,
    estimate_processing_time
)

# Define what gets exported when using "from ml_scripts.preprocessing.ocr import *"
__all__ = [
    # Main OCR class
    'TesseractOCR',
    
    # Quick functions
    'quick_ocr',
    'medical_document_ocr',
    
    # Configuration and validation
    'configure_tesseract',
    'validate_tesseract_installation',
    
    # File operations
    'find_images_in_directory',
    'get_supported_image_formats',
    
    # Configuration helpers
    'create_ocr_config',
    'get_medical_ocr_config',
    
    # Utilities
    'estimate_processing_time'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "EdgeMedicalSummarizer Team"
__description__ = "Tesseract OCR utilities for medical document processing"

# Quick setup function for convenience
def setup_ocr(tesseract_path: str = None, language: str = 'eng') -> TesseractOCR:
    """
    Quick setup function to initialize OCR with common settings.
    
    Args:
        tesseract_path: Optional path to tesseract executable
        language: Language code for OCR (default: 'eng')
        
    Returns:
        Configured TesseractOCR instance
    """
    if tesseract_path:
        configure_tesseract(tesseract_path)
    
    return TesseractOCR(language=language)

# Add setup function to exports
__all__.append('setup_ocr')