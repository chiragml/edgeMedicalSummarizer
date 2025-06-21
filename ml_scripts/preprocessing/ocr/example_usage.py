"""
Example Usage of Tesseract OCR Utility

This module demonstrates how to use the TesseractOCR class and utility functions.
"""

import os
from pathlib import Path
from tesseract_ocr import TesseractOCR, quick_ocr, medical_document_ocr
from utils import validate_tesseract_installation, find_images_in_directory


def example_basic_ocr():
    """Example of basic OCR usage."""
    print("=== Basic OCR Example ===")
    
    # Initialize OCR
    ocr = TesseractOCR(language='eng')
    
    # Example image path (you would replace this with an actual image)
    image_path = "sample_image.jpg"
    
    if os.path.exists(image_path):
        result = ocr.extract_text_from_image(image_path)
        
        if result['success']:
            print(f"Extracted Text: {result['text']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Word Count: {result['word_count']}")
        else:
            print(f"OCR failed: {result['error']}")
    else:
        print(f"Sample image not found: {image_path}")


def example_preprocessed_ocr():
    """Example of OCR with preprocessing."""
    print("\n=== Preprocessed OCR Example ===")
    
    ocr = TesseractOCR()
    
    # Preprocessing options
    preprocessing_options = {
        'grayscale': True,
        'enhance_contrast': True,
        'contrast_factor': 1.5,
        'enhance_sharpness': True,
        'sharpness_factor': 2.0,
        'resize': True,
        'scale_factor': 2.0
    }
    
    image_path = "sample_document.jpg"
    
    if os.path.exists(image_path):
        result = ocr.extract_text_from_image(
            image_path,
            preprocessing_options=preprocessing_options
        )
        
        if result['success']:
            print(f"Preprocessed OCR Result:")
            print(f"Text: {result['text'][:200]}...")  # First 200 characters
            print(f"Confidence: {result['confidence']}%")
        else:
            print(f"Preprocessed OCR failed: {result['error']}")
    else:
        print(f"Sample document not found: {image_path}")


def example_opencv_preprocessing():
    """Example of OCR with OpenCV preprocessing."""
    print("\n=== OpenCV Preprocessing Example ===")
    
    ocr = TesseractOCR()
    image_path = "noisy_document.jpg"
    
    if os.path.exists(image_path):
        result = ocr.extract_text_from_image(
            image_path,
            use_opencv_preprocessing=True
        )
        
        if result['success']:
            print(f"OpenCV Preprocessed Result:")
            print(f"Text: {result['text'][:200]}...")
            print(f"Confidence: {result['confidence']}%")
        else:
            print(f"OpenCV preprocessing failed: {result['error']}")
    else:
        print(f"Sample noisy document not found: {image_path}")


def example_bounding_boxes():
    """Example of extracting text with bounding boxes."""
    print("\n=== Bounding Boxes Example ===")
    
    ocr = TesseractOCR()
    image_path = "sample_image.jpg"
    
    if os.path.exists(image_path):
        result = ocr.extract_text_with_bounding_boxes(image_path)
        
        if result['success']:
            print(f"Found {result['total_words']} words")
            print(f"Full text: {result['full_text'][:100]}...")
            
            # Show first few words with their positions
            for i, word in enumerate(result['words'][:5]):
                print(f"Word {i+1}: '{word['text']}' at ({word['left']}, {word['top']}) "
                      f"confidence: {word['confidence']}%")
        else:
            print(f"Bounding box extraction failed: {result['error']}")
    else:
        print(f"Sample image not found: {image_path}")


def example_batch_processing():
    """Example of batch processing multiple images."""
    print("\n=== Batch Processing Example ===")
    
    ocr = TesseractOCR()
    
    # Create a sample directory with images (you would use an actual directory)
    image_directory = "sample_images"
    
    if os.path.exists(image_directory):
        result = ocr.batch_process_images(
            image_directory,
            output_file="batch_ocr_results.json"
        )
        
        if result['success']:
            print(f"Batch processing completed:")
            print(f"Successfully processed: {result['total_processed']} images")
            print(f"Failed: {result['total_failed']} images")
            
            # Show results for first few images
            for i, img_result in enumerate(result['results'][:3]):
                print(f"\nImage {i+1}: {img_result['filename']}")
                if img_result['success']:
                    print(f"  Text preview: {img_result['text'][:50]}...")
                    print(f"  Confidence: {img_result['confidence']}%")
                else:
                    print(f"  Error: {img_result['error']}")
        else:
            print(f"Batch processing failed: {result['error']}")
    else:
        print(f"Sample images directory not found: {image_directory}")


def example_quick_functions():
    """Example of using convenience functions."""
    print("\n=== Quick Functions Example ===")
    
    image_path = "sample_image.jpg"
    
    if os.path.exists(image_path):
        # Quick OCR
        text = quick_ocr(image_path)
        print(f"Quick OCR result: {text[:100]}...")
        
        # Medical document OCR
        medical_result = medical_document_ocr(image_path)
        if medical_result['success']:
            print(f"Medical OCR confidence: {medical_result['confidence']}%")
        else:
            print(f"Medical OCR failed: {medical_result['error']}")
    else:
        print(f"Sample image not found: {image_path}")


def validate_setup():
    """Validate that Tesseract is properly installed."""
    print("=== Tesseract Installation Validation ===")
    
    validation = validate_tesseract_installation()
    
    if validation['installed']:
        print(f"✓ Tesseract is installed")
        print(f"  Version: {validation['version']}")
        print(f"  Available languages: {', '.join(validation['available_languages'][:10])}...")
        print(f"  Default language: {validation['default_language']}")
    else:
        print(f"✗ Tesseract installation issue:")
        print(f"  Error: {validation['error']}")
        print(f"  Suggestion: {validation['suggestion']}")


def main():
    """Run all examples."""
    print("Tesseract OCR Utility Examples")
    print("=" * 50)
    
    # First validate the setup
    validate_setup()
    
    # Run examples
    example_basic_ocr()
    example_preprocessed_ocr()
    example_opencv_preprocessing()
    example_bounding_boxes()
    example_batch_processing()
    example_quick_functions()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: Most examples require actual image files to work.")
    print("Replace the sample image paths with real images to test the functionality.")


if __name__ == "__main__":
    main()