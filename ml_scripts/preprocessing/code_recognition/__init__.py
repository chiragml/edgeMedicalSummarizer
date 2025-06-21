"""
Code Recognition Module

This module handles recognition and structuring of medical codes from OCR output
using regular expression patterns and machine learning models.

Main Components:
- MedicalCode: Pydantic model for storing extracted codes
- ExtractionResult: Pydantic model for extraction results
- CodeExtractor: Main extraction class
- Pattern definitions: JSON-based code patterns
"""

# Import main models
from .models import (
    MedicalCode,
    ExtractionResult,
    CodeType,
    CodePattern
)

# Import extraction functionality
from .extraction import (
    CodeExtractor,
    extract_medical_codes,
    quick_extract_codes
)

# Define what gets exported when using "from ml_scripts.preprocessing.code_recognition import *"
__all__ = [
    # Models
    'MedicalCode',
    'ExtractionResult',
    'CodeType',
    'CodePattern',
    
    # Extraction classes and functions
    'CodeExtractor',
    'extract_medical_codes',
    'quick_extract_codes'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "EdgeMedicalSummarizer Team"
__description__ = "Medical billing code recognition and extraction utilities"

# Quick setup function for convenience
def setup_code_extractor(patterns_file: str = None) -> CodeExtractor:
    """
    Quick setup function to initialize code extractor.
    
    Args:
        patterns_file: Optional path to custom patterns file
        
    Returns:
        Configured CodeExtractor instance
    """
    return CodeExtractor(patterns_file)

# Add setup function to exports
__all__.append('setup_code_extractor')