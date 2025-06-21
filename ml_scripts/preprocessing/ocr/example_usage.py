"""
Example Usage of Tesseract OCR Utility

This module demonstrates how to use the OCR package with simplified imports.
"""

import os
from pathlib import Path

# Import directly from the OCR package
from ml_scripts.preprocessing.ocr import (
    TesseractOCR,
    quick_ocr,
    medical_document_ocr,
    configure_tesseract,
    validate_tesseract_installation,
    find_images_in_directory,
    setup_ocr,
    get_medical_ocr_config,
    estimate_processing_time
)

# Configure pytesseract to use the correct Tesseract path
configure_tesseract()


def example_basic_ocr():
    """Example of basic OCR usage with simplified setup."""
    print("=== Basic OCR Example ===")
    
    # Use the quick setup function
    ocr = setup_ocr(language='eng')
    
    # Example image path (you would replace this with an actual image)
    image_path = "C:\\Users\\HP\\Pictures\\bcc-chirag.png"
    
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


def example_quick_ocr():
    """Example using the quick_ocr function."""
    print("\n=== Quick OCR Example ===")
    
    image_path = "C:\\Users\\HP\\Pictures\\bcc-chirag.png"
    
    if os.path.exists(image_path):
        # Single line OCR extraction
        text = quick_ocr(image_path)
        print(f"Quick OCR Result: {text[:200]}...")
    else:
        print(f"Sample image not found: {image_path}")


def example_medical_ocr():
    """Example using medical document OCR."""
    print("\n=== Medical Document OCR Example ===")
    
    image_path = "C:\\Users\\HP\\Pictures\\bcc-chirag.png"
    
    if os.path.exists(image_path):
        result = medical_document_ocr(image_path)
        
        if result['success']:
            print(f"Medical OCR Result:")
            print(f"Text: {result['text'][:200]}...")
            print(f"Confidence: {result['confidence']}%")
            print(f"Preprocessing used: {result['preprocessing_used']}")
        else:
            print(f"Medical OCR failed: {result['error']}")
    else:
        print(f"Sample medical document not found: {image_path}")


def example_preprocessed_ocr():
    """Example of OCR with custom preprocessing."""
    print("\n=== Custom Preprocessed OCR Example ===")
    
    ocr = TesseractOCR()
    
    # Custom preprocessing options
    preprocessing_options = {
        'grayscale': True,
        'enhance_contrast': True,
        'contrast_factor': 1.5,
        'enhance_sharpness': True,
        'sharpness_factor': 2.0,
        'resize': True,
        'scale_factor': 2.0
    }
    
    image_path = "C:\\Users\\HP\\Pictures\\bcc-chirag.png"
    
    if os.path.exists(image_path):
        result = ocr.extract_text_from_image(
            image_path,
            preprocessing_options=preprocessing_options
        )
        
        if result['success']:
            print(f"Custom Preprocessed OCR Result:")
            print(f"Text: {result['text'][:200]}...")
            print(f"Confidence: {result['confidence']}%")
        else:
            print(f"Custom preprocessed OCR failed: {result['error']}")
    else:
        print(f"Sample document not found: {image_path}")


def example_opencv_preprocessing():
    """Example of OCR with OpenCV preprocessing."""
    print("\n=== OpenCV Preprocessing Example ===")
    
    ocr = TesseractOCR()
    image_path = "C:\\Users\\HP\\Pictures\\Screenshot 2025-01-09 161302.png"
    
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
    image_path = "C:\\Users\\HP\\Pictures\\bcc-chirag.png"
    
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
        # Estimate processing time first
        images = find_images_in_directory(image_directory)
        if images:
            time_estimate = estimate_processing_time(len(images))
            print(f"Found {len(images)} images")
            print(f"Estimated processing time: {time_estimate['total_estimated_time_minutes']:.1f} minutes")
        
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
        print("You can create a directory with sample images to test batch processing")


def example_configuration():
    """Example of using configuration helpers."""
    print("\n=== Configuration Example ===")
    
    # Get medical OCR configuration
    medical_config = get_medical_ocr_config()
    print(f"Medical OCR Config: {medical_config}")
    
    # Use custom configuration
    ocr = TesseractOCR()
    image_path = "C:\\Users\\HP\\Pictures\\bcc-chirag.png"
    
    if os.path.exists(image_path):
        result = ocr.extract_text_from_image(
            image_path,
            custom_config=medical_config
        )
        
        if result['success']:
            print(f"Medical config OCR result:")
            print(f"Text: {result['text'][:100]}...")
            print(f"Confidence: {result['confidence']}%")
        else:
            print(f"Medical config OCR failed: {result['error']}")
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
    """Run all examples with simplified imports."""
    print("Tesseract OCR Utility Examples (Simplified Imports)")
    print("=" * 60)
    
    # First validate the setup
    validate_setup()
    
    # Run examples
    example_basic_ocr()
    example_quick_ocr()
    example_medical_ocr()
    example_preprocessed_ocr()
    example_opencv_preprocessing()
    example_bounding_boxes()
    example_batch_processing()
    example_configuration()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nNote: Most examples require actual image files to work.")
    print("Replace the sample image paths with real images to test the functionality.")
    print("\nNow you can import OCR functions directly:")
    print("  from ml_scripts.preprocessing.ocr import TesseractOCR, quick_ocr")
    print("  from ml_scripts.preprocessing.ocr import setup_ocr, medical_document_ocr")


if __name__ == "__main__":
    main()