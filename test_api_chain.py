"""
API Test Script for Code Extraction

This script directly tests the /api/code-extract endpoint by sending it
a sample block of medical text and printing the structured data that
is returned. It uses a sample text from the code_recognition module's
example_usage.py file for a realistic test.
"""

import requests
import json

# --- Configuration ---
BASE_URL = "http://localhost:5000"
CODE_EXTRACT_ENDPOINT = f"{BASE_URL}/api/code-extract"

# --- Test Functions ---

def call_code_extract_api(text: str) -> dict | None:
    """Calls the code extraction endpoint and returns the result."""
    print("1. Calling Code Extraction API with sample text...")
    if not text or not text.strip():
        print("   [ERROR] Input text is empty. Skipping code extraction.")
        return None

    headers = {'Content-Type': 'application/json'}
    payload = {'text': text}

    try:
        response = requests.post(CODE_EXTRACT_ENDPOINT, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        response_data = response.json()
        if response_data.get('success'):
            print("   [SUCCESS] Code extraction successful.")
            return response_data.get('result')
        else:
            print(f"   [ERROR] Code Extraction API returned an error: {response_data.get('error')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] Failed to connect to the Code Extraction API: {e}")
        return None

def main():
    """Main function to run the code extraction API test."""
    print("--- Starting API Test: Code Extraction ---")
    
    # Sample medical text taken from the code_recognition example_usage.py
    sample_text = """
    Patient diagnosed with diabetes mellitus type 2 (ICD-10: E11.9).
    Performed office visit (CPT 99213) and glucose test (CPT 80053).
    Prescribed medication NDC: 0069-2587-68.
    DRG: 470 assigned for billing.
    Lab test LOINC: 33747-0 ordered.
    """
    
    # Call the code extraction API with the sample text
    extraction_result = call_code_extract_api(sample_text)
    
    if extraction_result:
        print("\n--- Final Extraction Result ---")
        # Use json.dumps for pretty printing the final dictionary
        print(json.dumps(extraction_result, indent=2))
    
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()