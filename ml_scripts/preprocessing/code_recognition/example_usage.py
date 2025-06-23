"""
Example Usage of Medical Code Recognition

This module demonstrates how to use the code recognition package.
"""


from . import (
    CodeExtractor,
    MedicalCode,
    ExtractionResult,
    CodeType,
    extract_medical_codes,
    quick_extract_codes,
    setup_code_extractor
)

def example_basic_extraction():
    """Example of basic code extraction."""
    print("=== Basic Code Extraction Example ===")
    
    # Sample medical text with various codes
    sample_text = """
    Patient diagnosed with diabetes mellitus type 2 (ICD-10: E11.9).
    Performed office visit (CPT 99213) and glucose test (CPT 80053).
    Prescribed medication NDC: 0069-2587-68.
    DRG: 470 assigned for billing.
    Lab test LOINC: 33747-0 ordered.
    """
    
    # Initialize extractor
    extractor = CodeExtractor()
    
    # Extract codes
    result = extractor.extract_codes(sample_text)
    
    print(f"Input text: {sample_text.strip()}")
    print(f"\nExtracted {result.total_codes_found} codes:")
    
    for i, code in enumerate(result.extracted_codes, 1):
        # print(f"Code: {result}")
        print(f"{i}. {code.code_type}: {code.code_value}")
        print(f"   Original text: '{code.original_text}'")
        print(f"   Position: {code.start_position}-{code.end_position}")
        print(f"   Context: {code.context[:100]}..." if code.context else "")
        print()


def example_specific_code_types():
    """Example of extracting specific code types."""
    print("=== Specific Code Types Example ===")
    
    sample_text = """
    Patient visit for routine checkup.
    ICD-10 codes: E11.9, Z00.00, I10
    CPT codes: 99213, 36415, 80053
    HCPCS: A0425, J1100
    """
    
    extractor = CodeExtractor()
    
    # Extract only ICD-10-CM codes
    icd_codes = extractor.extract_codes_by_type(sample_text, CodeType.ICD10_CM)
    print("ICD-10-CM codes found:")
    for code in icd_codes:
        print(f"  - {code.code_value}")
    
    # Extract only CPT codes
    cpt_codes = extractor.extract_codes_by_type(sample_text, CodeType.CPT)
    print("\nCPT codes found:")
    for code in cpt_codes:
        print(f"  - {code.code_value}")


def example_validation():
    """Example of code validation."""
    print("\n=== Code Validation Example ===")
    
    extractor = CodeExtractor()
    
    # Test various codes
    test_codes = [
        ("E11.9", CodeType.ICD10_CM),
        ("99213", CodeType.CPT),
        ("A0425", CodeType.HCPCS),
        ("INVALID", CodeType.CPT),
        ("123", CodeType.ICD10_CM)
    ]
    
    print("Code validation results:")
    for code_value, code_type in test_codes:
        is_valid = extractor.validate_code(code_value, code_type)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"  {code_value} as {code_type.value}: {status}")


def example_pattern_info():
    """Example of getting pattern information."""
    print("\n=== Pattern Information Example ===")
    
    extractor = CodeExtractor()
    
    # Get pattern info for CPT codes
    cpt_patterns = extractor.get_pattern_info(CodeType.CPT)
    print("CPT Code Patterns:")
    for i, pattern in enumerate(cpt_patterns, 1):
        print(f"  {i}. Pattern: {pattern['pattern']}")
        print(f"     Description: {pattern['description']}")
        print(f"     Examples: {', '.join(pattern['examples'])}")
        print(f"     Priority: {pattern['priority']}")
        print()


def example_convenience_functions():
    """Example of using convenience functions."""
    print("=== Convenience Functions Example ===")
    
    sample_text = "Patient has diabetes E11.9 and hypertension I10. Office visit 99213."
    
    # Quick extraction with full result
    result = extract_medical_codes(sample_text)
    print(f"Full extraction found {result.total_codes_found} codes")
    
    # Quick extraction with just code values
    codes = quick_extract_codes(sample_text)
    print(f"Quick extraction: {codes}")


def example_result_analysis():
    """Example of analyzing extraction results."""
    print("\n=== Result Analysis Example ===")
    
    sample_text = """
    Encounter for diabetes management.
    Primary diagnosis: E11.9 (Type 2 diabetes)
    Secondary: I10 (Hypertension), Z79.4 (Long term insulin use)
    Procedures: 99213 (Office visit), 80053 (Glucose test), 36415 (Blood draw)
    Medications: NDC 0069-2587-68, NDC 54868-4313-0
    """
    
    result = extract_medical_codes(sample_text)
    
    # Get summary
    summary = result.to_summary()
    print("Extraction Summary:")
    print(f"  Total codes: {summary['total_codes']}")
    print(f"  Unique codes: {summary['unique_codes']}")
    print(f"  Processing time: {summary['processing_time']} seconds")
    print(f"  Codes by type: {summary['codes_by_type']}")
    
    # Get codes by type
    print("\nCodes by type:")
    for code_type in CodeType:
        type_codes = result.get_codes_by_type(code_type)
        if type_codes:
            print(f"  {code_type}: {[c.code_value for c in type_codes]}")
    
    # Get unique codes
    unique_codes = result.get_unique_codes()
    print(f"\nUnique codes: {[f'{c.code_type}:{c.code_value}' for c in unique_codes]}")


def example_custom_patterns():
    """Example of using custom patterns file."""
    print("\n=== Custom Patterns Example ===")
    
    # This would use a custom patterns file if available
    try:
        custom_extractor = CodeExtractor("custom_patterns.json")
        print("Custom patterns loaded successfully")
    except FileNotFoundError:
        print("Custom patterns file not found, using default patterns")
        custom_extractor = CodeExtractor()
    
    # Show supported code types
    supported_types = custom_extractor.get_supported_code_types()
    print(f"Supported code types: {[ct.value for ct in supported_types]}")


def example_setup_function():
    """Example of using the setup function."""
    print("\n=== Setup Function Example ===")
    
    # Quick setup
    extractor = setup_code_extractor()
    
    sample_text = "Patient visit CPT 99213 for diabetes E11.9"
    result = extractor.extract_codes(sample_text)
    
    print(f"Setup function created extractor that found {result.total_codes_found} codes")


def main():
    """Run all examples."""
    print("Medical Code Recognition Examples")
    print("=" * 50)
    
    example_basic_extraction()
    example_specific_code_types()
    example_validation()
    example_pattern_info()
    example_convenience_functions()
    example_result_analysis()
    example_custom_patterns()
    example_setup_function()
    
    print("=" * 50)
    print("Examples completed!")
    print("\nNow you can use code recognition in your applications:")
    print("  from ml_scripts.preprocessing.code_recognition import CodeExtractor")
    print("  extractor = CodeExtractor()")
    print("  result = extractor.extract_codes(your_text)")


if __name__ == "__main__":
    main()