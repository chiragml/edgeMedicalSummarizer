"""
Flask API for Medical Document Processing

This application provides two main endpoints:
1. /api/ocr-extract: To perform OCR on an uploaded image file.
2. /api/code-extract: To extract structured codes from text using regex.

It is designed to be run on an edge device.
"""

import os
import logging
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
import pytesseract
from PIL import Image
import io

# --- Import your custom ML/processing modules ---
# The OCR module is already built and located in the ml_scripts directory
from ml_scripts.preprocessing.ocr import validate_tesseract_installation, configure_tesseract

# Using your existing module for code extraction from the 'code_recognition' module.
from ml_scripts.preprocessing.code_recognition import extract_medical_codes

# --- Pydantic Models for Request Validation ---
class CodeExtractRequest(BaseModel):
    text: str


# --- Basic Flask App Setup ---
app = Flask(__name__)

# Create a directory for temporary file uploads
UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define allowed file extensions for OCR to prevent errors with invalid file types.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}

def allowed_file(filename):
    """Checks if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Dependency Validation at Startup ---
@app.before_request
def check_tesseract():
    # This check runs once before the first request
    if not hasattr(app, 'tesseract_checked'):
        logging.info("Performing one-time Tesseract installation check...")
        tesseract_status = validate_tesseract_installation()
        if not tesseract_status['installed']:
            logging.error("--- TESSERACT OCR IS NOT INSTALLED OR CONFIGURED CORRECTLY ---")
            logging.error(f"Error: {tesseract_status.get('error')}")
            logging.error(f"Suggestion: {tesseract_status.get('suggestion')}")
        else:
            # Ensure pytesseract knows the path if not in system PATH.
            # This is redundant if Tesseract is in PATH, but safe to do.
            configure_tesseract()
            logging.info("Tesseract installation check passed.")
        app.tesseract_checked = True


# --- API Endpoints ---

@app.route('/api/ocr-extract', methods=['POST'])
def ocr_extract_endpoint():
    """
    Endpoint to extract text from an image file sent in the request.
    The image is processed in-memory without being saved to disk.
    Usage: curl -X POST -F "file=@/path/to/your/image.png" http://localhost:5000/api/ocr-extract
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected for uploading'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': f'Invalid file type. Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        logging.info(f"File '{file.filename}' received. Starting in-memory OCR process.")
        
        # Read the image file into a memory buffer
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Perform OCR directly on the in-memory image object
        extracted_text = pytesseract.image_to_string(pil_image)

        if not extracted_text.strip():
            return jsonify({'success': False, 'error': 'OCR failed to extract text. The image might be empty or unreadable.'}), 500

        return jsonify({'success': True, 'extracted_text': extracted_text})
    except Exception as e:
        logging.error(f"An error occurred during OCR processing: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'An internal server error occurred: {str(e)}'}), 500

@app.route('/api/code-extract', methods=['POST'])
def code_extract_endpoint():
    """
    Endpoint to extract structured medical codes from a block of text.
    Usage: curl -X POST -H "Content-Type: application/json" -d '{"text": "The patient was diagnosed with I10 and E11.9."}' http://localhost:5000/api/code-extract
    """
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Invalid content type, must be application/json'}), 415

    try:
        # Validate the request body against the Pydantic model
        req_data = CodeExtractRequest.model_validate(request.get_json())
    except ValidationError as e:
        # Return a detailed validation error from Pydantic
        return jsonify({'success': False, 'error': 'Invalid request payload', 'details': e.errors()}), 400
    except Exception:
        # Catch potential malformed JSON errors
        return jsonify({'success': False, 'error': 'Malformed JSON in request body'}), 400

    logging.info("Received text for code extraction.")
    # Use the full extraction function to get detailed results.
    # The result is a Pydantic model, so we convert it to a JSON-serializable dict.
    extraction_result = extract_medical_codes(req_data.text)
    
    # .model_dump(mode='json') is used to ensure Pydantic objects (like Enums) are properly serialized.
    response_data = extraction_result.model_dump(mode='json')
    return jsonify({'success': True, 'result': response_data})

if __name__ == '__main__':
    # For development only. Use a production-ready WSGI server like Gunicorn or Waitress for deployment.
    app.run(host='0.0.0.0', port=5000, debug=True)
