"""
Tesseract OCR Utility

This module provides OCR functionality using Tesseract with image preprocessing
and support for multiple image formats.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TesseractOCR:
    """
    A comprehensive OCR utility using Tesseract with preprocessing capabilities.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, language: str = 'eng'):
        """
        Initialize the TesseractOCR instance.
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
            language: Language code for OCR (default: 'eng')
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.language = language
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    
    def validate_image_path(self, image_path: str) -> bool:
        """
        Validate if the image path exists and has a supported format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if valid, False otherwise
        """
        path = Path(image_path)
        
        if not path.exists():
            logger.error(f"Image file does not exist: {image_path}")
            return False
        
        if path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported image format: {path.suffix}")
            return False
        
        return True
    
    def preprocess_image(self, image: Image.Image, preprocessing_options: Dict[str, Any]) -> Image.Image:
        """
        Apply preprocessing to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            preprocessing_options: Dictionary of preprocessing options
            
        Returns:
            PIL Image object after preprocessing
        """
        processed_image = image.copy()
        
        # Convert to grayscale if specified
        if preprocessing_options.get('grayscale', False):
            processed_image = processed_image.convert('L')
        
        # Enhance contrast
        if preprocessing_options.get('enhance_contrast', False):
            enhancer = ImageEnhance.Contrast(processed_image)
            contrast_factor = preprocessing_options.get('contrast_factor', 1.5)
            processed_image = enhancer.enhance(contrast_factor)
        
        # Enhance sharpness
        if preprocessing_options.get('enhance_sharpness', False):
            enhancer = ImageEnhance.Sharpness(processed_image)
            sharpness_factor = preprocessing_options.get('sharpness_factor', 2.0)
            processed_image = enhancer.enhance(sharpness_factor)
        
        # Apply filters
        if preprocessing_options.get('apply_filter'):
            filter_type = preprocessing_options.get('filter_type', 'SHARPEN')
            if hasattr(ImageFilter, filter_type):
                processed_image = processed_image.filter(getattr(ImageFilter, filter_type))
        
        # Resize image if specified
        if preprocessing_options.get('resize'):
            scale_factor = preprocessing_options.get('scale_factor', 2.0)
            new_size = (
                int(processed_image.width * scale_factor),
                int(processed_image.height * scale_factor)
            )
            processed_image = processed_image.resize(new_size, Image.Resampling.LANCZOS)
        
        return processed_image
    
    def advanced_preprocess_with_opencv(self, image_path: str) -> np.ndarray:
        """
        Apply advanced preprocessing using OpenCV for better OCR results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image with OpenCV
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        return processed
    
    def extract_text_from_image(
        self,
        image_path: str,
        preprocessing_options: Optional[Dict[str, Any]] = None,
        use_opencv_preprocessing: bool = False,
        custom_config: str = ''
    ) -> Dict[str, Any]:
        """
        Extract text from an image using Tesseract OCR.
        
        Args:
            image_path: Path to the image file
            preprocessing_options: Dictionary of preprocessing options
            use_opencv_preprocessing: Whether to use OpenCV for advanced preprocessing
            custom_config: Custom Tesseract configuration string
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Validate image path
            if not self.validate_image_path(image_path):
                return {
                    'success': False,
                    'error': 'Invalid image path or unsupported format',
                    'text': '',
                    'confidence': 0
                }
            
            # Load image
            if use_opencv_preprocessing:
                # Use OpenCV preprocessing
                processed_array = self.advanced_preprocess_with_opencv(image_path)
                image = Image.fromarray(processed_array)
            else:
                # Use PIL preprocessing
                image = Image.open(image_path)
                if preprocessing_options:
                    image = self.preprocess_image(image, preprocessing_options)
            
            # Configure Tesseract
            config = f'--oem 3 --psm 6 -l {self.language}'
            if custom_config:
                config += f' {custom_config}'
            
            # Extract text
            extracted_text = pytesseract.image_to_string(image, config=config)
            
            # Get confidence scores
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'success': True,
                'text': extracted_text.strip(),
                'confidence': round(avg_confidence, 2),
                'word_count': len(extracted_text.split()),
                'image_path': image_path,
                'preprocessing_used': preprocessing_options is not None or use_opencv_preprocessing
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0
            }
    
    def extract_text_with_bounding_boxes(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text with bounding box information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing text and bounding box data
        """
        try:
            if not self.validate_image_path(image_path):
                return {'success': False, 'error': 'Invalid image path'}
            
            image = Image.open(image_path)
            
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(
                image,
                config=f'--oem 3 --psm 6 -l {self.language}',
                output_type=pytesseract.Output.DICT
            )
            
            # Process the data to extract meaningful information
            words_data = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Filter out low confidence detections
                    word_info = {
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                    words_data.append(word_info)
            
            return {
                'success': True,
                'words': words_data,
                'full_text': ' '.join([word['text'] for word in words_data]),
                'total_words': len(words_data)
            }
            
        except Exception as e:
            logger.error(f"Bounding box extraction failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def batch_process_images(self, image_directory: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple images in a directory.
        
        Args:
            image_directory: Path to directory containing images
            output_file: Optional path to save results
            
        Returns:
            Dictionary containing batch processing results
        """
        try:
            directory_path = Path(image_directory)
            if not directory_path.exists():
                return {'success': False, 'error': 'Directory does not exist'}
            
            results = []
            processed_count = 0
            failed_count = 0
            
            # Process each image in the directory
            for image_file in directory_path.iterdir():
                if image_file.suffix.lower() in self.supported_formats:
                    result = self.extract_text_from_image(str(image_file))
                    result['filename'] = image_file.name
                    results.append(result)
                    
                    if result['success']:
                        processed_count += 1
                    else:
                        failed_count += 1
            
            # Save results if output file specified
            if output_file and results:
                self._save_batch_results(results, output_file)
            
            return {
                'success': True,
                'total_processed': processed_count,
                'total_failed': failed_count,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _save_batch_results(self, results: list, output_file: str) -> None:
        """Save batch processing results to a file."""
        import json
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Batch results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save batch results: {str(e)}")


# Convenience functions for easy usage
def quick_ocr(image_path: str, language: str = 'eng') -> str:
    """
    Quick OCR extraction with default settings.
    
    Args:
        image_path: Path to the image file
        language: Language code for OCR
        
    Returns:
        Extracted text as string
    """
    ocr = TesseractOCR(language=language)
    result = ocr.extract_text_from_image(image_path)
    return result.get('text', '') if result['success'] else ''


def medical_document_ocr(image_path: str) -> Dict[str, Any]:
    """
    OCR optimized for medical documents with preprocessing.
    
    Args:
        image_path: Path to the medical document image
        
    Returns:
        Dictionary containing OCR results and metadata
    """
    ocr = TesseractOCR(language='eng')
    
    # Preprocessing options optimized for medical documents
    preprocessing_options = {
        'grayscale': True,
        'enhance_contrast': True,
        'contrast_factor': 1.3,
        'enhance_sharpness': True,
        'sharpness_factor': 1.5,
        'resize': True,
        'scale_factor': 1.5
    }
    
    return ocr.extract_text_from_image(
        image_path,
        preprocessing_options=preprocessing_options,
        custom_config='--psm 6'  # Assume uniform block of text
    )