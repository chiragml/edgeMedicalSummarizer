# Medical Code Recognition Module

A comprehensive Python module for extracting, validating, and processing medical billing codes from unstructured text using pattern matching and regular expressions.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture & Design Patterns](#architecture--design-patterns)
- [Supported Code Types](#supported-code-types)
- [Installation & Setup](#installation--setup)
- [Core Components](#core-components)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Pattern Configuration](#pattern-configuration)
- [Performance Considerations](#performance-considerations)
- [Error Handling](#error-handling)
- [Contributing](#contributing)

## 🎯 Overview

The Medical Code Recognition module is designed to extract structured medical billing codes from unstructured text documents, such as medical reports, insurance claims, and clinical notes. It uses a combination of regular expression patterns, validation logic, and configurable extraction rules to identify and categorize various types of medical codes.

### Key Features

- **Multi-Code Type Support**: Extracts 13 different medical code types
- **Pattern-Based Extraction**: JSON-configurable regex patterns
- **Validation & Confidence Scoring**: Built-in code validation and confidence assessment
- **Context Preservation**: Maintains original text context and positioning
- **Flexible API**: Multiple extraction methods for different use cases
- **Performance Optimized**: Compiled regex patterns and efficient algorithms

## 🏗️ Architecture & Design Patterns

### 1. **Factory Pattern**
The `CodeExtractor` class acts as a factory for creating extraction instances with different pattern configurations.

### 2. **Strategy Pattern**
Different extraction strategies are implemented through configurable regex patterns, allowing easy addition of new code types without modifying core logic.

### 3. **Builder Pattern**
The `MedicalCode` and `ExtractionResult` models use Pydantic's builder pattern for type-safe object construction with validation.

### 4. **Template Method Pattern**
The extraction process follows a consistent template: pattern loading → compilation → extraction → validation → result assembly.

### 5. **Observer Pattern**
Logging is integrated throughout the extraction process to monitor performance and debug issues.

## 🏥 Supported Code Types

| Code Type | Description | Example | Priority |
|-----------|-------------|---------|----------|
| **ICD-10-CM** | Diagnosis codes | `E11.9`, `Z00.00` | High |
| **ICD-10-PCS** | Procedure codes | `0016070`, `00160J6` | Medium |
| **CPT** | Current Procedural Terminology | `99213`, `36415` | High |
| **HCPCS** | Healthcare Common Procedure Coding System | `A0425`, `J1100` | High |
| **DRG** | Diagnosis-Related Groups | `470`, `291` | Medium |
| **NDC** | National Drug Codes | `0069-2587-68` | High |
| **LOINC** | Logical Observation Identifiers | `33747-0`, `718-7` | Medium |
| **SNOMED-CT** | Systematized Nomenclature of Medicine | `386661006` | Low |
| **ICD-9-CM** | Legacy diagnosis codes | `250.00`, `401.9` | Low |
| **ICD-9-PCS** | Legacy procedure codes | `00.66`, `39.95` | Low |
| **MODIFIER** | CPT/HCPCS modifiers | `22`, `59`, `AA` | Medium |
| **REVENUE-CODE** | Revenue codes | `0450`, `0636` | Medium |
| **PLACE-OF-SERVICE** | Service location codes | `11`, `23` | Medium |

## 🚀 Installation & Setup

### Prerequisites
```bash
pip install pydantic
```

### Basic Setup
```python
from ml_scripts.preprocessing.code_recognition import CodeExtractor

# Initialize with default patterns
extractor = CodeExtractor()

# Or with custom patterns file
extractor = CodeExtractor("custom_patterns.json")
```

### Quick Setup Function
```python
from ml_scripts.preprocessing.code_recognition import setup_code_extractor

extractor = setup_code_extractor()
```

## 🔧 Core Components

### 1. **MedicalCode Model**
Represents an individual extracted medical code with metadata.

```python
from ml_scripts.preprocessing.code_recognition import MedicalCode, CodeType

code = MedicalCode(
    code_type=CodeType.ICD10_CM,
    code_value="E11.9",
    original_text="diabetes mellitus type 2 (E11.9)",
    start_position=25,
    end_position=30,
    confidence_score=0.95,
    context="Patient diagnosed with diabetes mellitus type 2 (E11.9)."
)
```

### 2. **ExtractionResult Model**
Contains the complete result of a code extraction operation.

```python
from ml_scripts.preprocessing.code_recognition import ExtractionResult

result = ExtractionResult(
    input_text="Patient visit for diabetes E11.9",
    extracted_codes=[code1, code2],
    processing_time_seconds=0.05
)
```

### 3. **CodeExtractor Class**
Main extraction engine with configurable patterns and validation.

```python
extractor = CodeExtractor()
result = extractor.extract_codes("Patient has diabetes E11.9")
```

## 📚 API Reference

### CodeExtractor Methods

#### `extract_codes(text, code_types=None, include_context=True, context_window=50)`
Extract all medical codes from text.

**Parameters:**
- `text` (str): Input text to process
- `code_types` (List[CodeType], optional): Specific code types to extract
- `include_context` (bool): Include surrounding context
- `context_window` (int): Characters around code for context

**Returns:** `ExtractionResult`

#### `extract_codes_by_type(text, code_type, include_context=True)`
Extract codes of a specific type only.

**Parameters:**
- `text` (str): Input text to process
- `code_type` (CodeType): Type of codes to extract
- `include_context` (bool): Include surrounding context

**Returns:** `List[MedicalCode]`

#### `validate_code(code_value, code_type)`
Validate if a code matches the expected format.

**Parameters:**
- `code_value` (str): Code to validate
- `code_type` (CodeType): Expected code type

**Returns:** `bool`

#### `get_pattern_info(code_type)`
Get information about patterns for a code type.

**Parameters:**
- `code_type` (CodeType): Code type to get patterns for

**Returns:** `List[Dict[str, Any]]`

#### `get_supported_code_types()`
Get list of all supported code types.

**Returns:** `List[CodeType]`

### Convenience Functions

#### `extract_medical_codes(text, patterns_file=None, code_types=None)`
Quick extraction with full result object.

#### `quick_extract_codes(text)`
Quick extraction returning only code values as strings.

## 💡 Usage Examples

### Basic Code Extraction

```python
from ml_scripts.preprocessing.code_recognition import CodeExtractor

# Initialize extractor
extractor = CodeExtractor()

# Sample medical text
text = """
Patient diagnosed with diabetes mellitus type 2 (ICD-10: E11.9).
Performed office visit (CPT 99213) and glucose test (CPT 80053).
Prescribed medication NDC: 0069-2587-68.
DRG: 470 assigned for billing.
"""

# Extract all codes
result = extractor.extract_codes(text)

print(f"Found {result.total_codes_found} codes:")
for code in result.extracted_codes:
    print(f"- {code.code_type}: {code.code_value}")
    print(f"  Context: {code.context}")
```

### Specific Code Type Extraction

```python
from ml_scripts.preprocessing.code_recognition import CodeType

# Extract only ICD-10-CM codes
icd_codes = extractor.extract_codes_by_type(text, CodeType.ICD10_CM)

# Extract only CPT codes
cpt_codes = extractor.extract_codes_by_type(text, CodeType.CPT)

print("ICD-10-CM codes:", [c.code_value for c in icd_codes])
print("CPT codes:", [c.code_value for c in cpt_codes])
```

### Code Validation

```python
# Validate individual codes
is_valid_icd = extractor.validate_code("E11.9", CodeType.ICD10_CM)
is_valid_cpt = extractor.validate_code("99213", CodeType.CPT)
is_invalid = extractor.validate_code("INVALID", CodeType.CPT)

print(f"E11.9 valid: {is_valid_icd}")
print(f"99213 valid: {is_valid_cpt}")
print(f"INVALID valid: {is_invalid}")
```

### Result Analysis

```python
# Get summary statistics
summary = result.to_summary()
print(f"Total codes: {summary['total_codes']}")
print(f"Unique codes: {summary['unique_codes']}")
print(f"Processing time: {summary['processing_time']} seconds")
print(f"Codes by type: {summary['codes_by_type']}")

# Get codes by type
for code_type in CodeType:
    type_codes = result.get_codes_by_type(code_type)
    if type_codes:
        print(f"{code_type}: {[c.code_value for c in type_codes]}")

# Get unique codes
unique_codes = result.get_unique_codes()
print(f"Unique codes: {[f'{c.code_type}:{c.code_value}' for c in unique_codes]}")
```

### Advanced Usage with Custom Patterns

```python
# Initialize with custom patterns file
custom_extractor = CodeExtractor("my_custom_patterns.json")

# Extract with specific code types only
target_types = [CodeType.ICD10_CM, CodeType.CPT, CodeType.NDC]
result = custom_extractor.extract_codes(
    text=medical_text,
    code_types=target_types,
    include_context=True,
    context_window=100
)
```

### Batch Processing

```python
# Process multiple documents
documents = [
    "Patient A: diabetes E11.9, visit 99213",
    "Patient B: hypertension I10, test 80053",
    "Patient C: asthma J45.909, medication NDC 0069-2587-68"
]

results = []
for doc in documents:
    result = extractor.extract_codes(doc)
    results.append(result)

# Aggregate results
total_codes = sum(r.total_codes_found for r in results)
print(f"Total codes across all documents: {total_codes}")
```

## ⚙️ Pattern Configuration

### Pattern File Structure

Patterns are defined in JSON format with the following structure:

```json
{
  "CODE_TYPE": [
    {
      "pattern": "regex_pattern",
      "description": "Human-readable description",
      "examples": ["example1", "example2"],
      "priority": 1,
      "case_sensitive": false
    }
  ]
}
```

### Pattern Properties

- **pattern**: Regular expression pattern for matching codes
- **description**: Human-readable description of the pattern
- **examples**: List of example codes that match this pattern
- **priority**: Matching priority (higher = more important)
- **case_sensitive**: Whether pattern matching is case-sensitive

### Creating Custom Patterns

```python
# Example custom pattern for a new code type
custom_patterns = {
    "CUSTOM_CODE": [
        {
            "pattern": r"\bCUST\d{4}\b",
            "description": "Custom 4-digit codes starting with CUST",
            "examples": ["CUST1234", "CUST5678"],
            "priority": 3,
            "case_sensitive": False
        }
    ]
}

# Save to file
import json
with open("custom_patterns.json", "w") as f:
    json.dump(custom_patterns, f, indent=2)
```

## ⚡ Performance Considerations

### Optimization Strategies

1. **Pattern Compilation**: All regex patterns are pre-compiled for faster matching
2. **Priority-Based Matching**: Higher priority patterns are processed first
3. **Overlap Prevention**: Avoids duplicate extractions at the same text positions
4. **Context Window Limiting**: Configurable context window to reduce memory usage

### Performance Tips

```python
# For high-volume processing, reuse extractor instance
extractor = CodeExtractor()

# Process in batches
batch_size = 1000
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    for doc in batch:
        result = extractor.extract_codes(doc)

# Disable context for faster processing if not needed
result = extractor.extract_codes(text, include_context=False)
```

### Memory Usage

- **Pattern Storage**: ~50KB for default patterns
- **Per Extraction**: ~1KB per extracted code
- **Context Storage**: ~200 bytes per code (with 50-char context)

## 🛡️ Error Handling

### Common Error Scenarios

```python
try:
    # Invalid patterns file
    extractor = CodeExtractor("nonexistent.json")
except FileNotFoundError:
    print("Patterns file not found")

try:
    # Invalid JSON in patterns file
    extractor = CodeExtractor("malformed.json")
except json.JSONDecodeError:
    print("Invalid JSON in patterns file")

try:
    # Invalid regex pattern
    extractor = CodeExtractor("invalid_patterns.json")
except re.error:
    print("Invalid regex pattern")
```

### Validation Errors

```python
try:
    # Invalid MedicalCode creation
    code = MedicalCode(
        code_type=CodeType.ICD10_CM,
        code_value="",  # Empty value will raise error
        original_text="test",
        start_position=0,
        end_position=4
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Graceful Degradation

```python
# Handle missing patterns gracefully
try:
    extractor = CodeExtractor("custom_patterns.json")
except FileNotFoundError:
    print("Using default patterns")
    extractor = CodeExtractor()

# Handle extraction errors
try:
    result = extractor.extract_codes(text)
except Exception as e:
    print(f"Extraction failed: {e}")
    result = ExtractionResult(input_text=text, extracted_codes=[])
```

## 🤝 Contributing

### Adding New Code Types

1. **Update CodeType Enum**:
```python
class CodeType(str, Enum):
    # ... existing types ...
    NEW_CODE_TYPE = "NEW-CODE-TYPE"
```

2. **Add Patterns**:
```json
{
  "NEW-CODE-TYPE": [
    {
      "pattern": "your_regex_pattern",
      "description": "Description of new code type",
      "examples": ["example1", "example2"],
      "priority": 2,
      "case_sensitive": false
    }
  ]
}
```

3. **Update Tests**: Add test cases for the new code type

### Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Add docstrings for all public methods
- Include examples in docstrings
- Write unit tests for new functionality

### Testing

```python
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=ml_scripts.preprocessing.code_recognition tests/
```

## 📄 License

This module is part of the EdgeMedicalSummarizer project. See the main project license for details.

## 🆘 Support

For issues, questions, or contributions:

1. Check the existing examples in `example_usage.py`
2. Review the pattern definitions in `code_patterns.json`
3. Examine the test cases for usage patterns
4. Create an issue with detailed error information

---

**Note**: This module is designed for educational and research purposes. For production use in healthcare applications, ensure compliance with relevant regulations and validate results against authoritative medical coding databases. This module tries to comply with HIPPA regulation as closely as possible.  