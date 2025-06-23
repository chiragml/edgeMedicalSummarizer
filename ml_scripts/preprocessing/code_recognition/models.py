"""
Pydantic Data Models for Medical Code Recognition

This module defines the data structures for storing extracted medical billing codes.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class CodeType(str, Enum):
    """Enumeration of medical code types."""
    ICD10_CM = "ICD-10-CM"
    ICD10_PCS = "ICD-10-PCS"
    CPT = "CPT"
    HCPCS = "HCPCS"
    DRG = "DRG"
    NDC = "NDC"
    LOINC = "LOINC"
    SNOMED_CT = "SNOMED-CT"
    ICD9_CM = "ICD-9-CM"
    ICD9_PCS = "ICD-9-PCS"
    MODIFIER = "MODIFIER"
    REVENUE_CODE = "REVENUE-CODE"
    PLACE_OF_SERVICE = "PLACE-OF-SERVICE"


class MedicalCode(BaseModel):
    """
    Pydantic model representing an extracted medical billing code.
    """
    code_type: CodeType = Field(..., description="Type of medical code")
    code_value: str = Field(..., description="The actual code value extracted")
    original_text: str = Field(..., description="Original text where code was found")
    start_position: int = Field(..., description="Start position in the original text")
    end_position: int = Field(..., description="End position in the original text")
    confidence_score: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the extraction (0.0 to 1.0)"
    )
    context: Optional[str] = Field(
        default=None, 
        description="Surrounding context of the extracted code"
    )
    description: Optional[str] = Field(
        default=None, 
        description="Human-readable description of the code if available"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata about the extraction"
    )

    @validator('code_value')
    def validate_code_value(cls, v):
        """Validate that code_value is not empty."""
        if not v or not v.strip():
            raise ValueError('code_value cannot be empty')
        return v.strip().upper()

    @validator('original_text')
    def validate_original_text(cls, v):
        """Validate that original_text is not empty."""
        if not v or not v.strip():
            raise ValueError('original_text cannot be empty')
        return v.strip()

    @validator('end_position')
    def validate_positions(cls, v, values):
        """Validate that end_position is greater than start_position."""
        if 'start_position' in values and v <= values['start_position']:
            raise ValueError('end_position must be greater than start_position')
        return v

    def __str__(self) -> str:
        return f"{self.code_type}: {self.code_value}"

    def __repr__(self) -> str:
        return f"MedicalCode(type={self.code_type}, code={self.code_value})"

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


class ExtractionResult(BaseModel):
    """
    Pydantic model representing the complete result of code extraction.
    """
    input_text: str = Field(..., description="Original input text")
    extracted_codes: List[MedicalCode] = Field(
        default_factory=list, 
        description="List of extracted medical codes"
    )
    total_codes_found: int = Field(
        default=0, 
        description="Total number of codes extracted"
    )
    extraction_timestamp: Optional[str] = Field(
        default=None, 
        description="Timestamp when extraction was performed"
    )
    processing_time_seconds: Optional[float] = Field(
        default=None, 
        description="Time taken for extraction in seconds"
    )
    extraction_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata about the extraction process"
    )

    @validator('total_codes_found', always=True)
    def validate_total_codes(cls, v, values):
        """Ensure total_codes_found matches the length of extracted_codes."""
        if 'extracted_codes' in values:
            return len(values['extracted_codes'])
        return v

    def get_codes_by_type(self, code_type: CodeType) -> List[MedicalCode]:
        """Get all codes of a specific type."""
        return [code for code in self.extracted_codes if code.code_type == code_type]

    def get_unique_codes(self) -> List[MedicalCode]:
        """Get unique codes (removing duplicates based on code_type and code_value)."""
        seen = set()
        unique_codes = []
        
        for code in self.extracted_codes:
            key = (code.code_type, code.code_value)
            if key not in seen:
                seen.add(key)
                unique_codes.append(code)
        
        return unique_codes

    def to_summary(self) -> Dict[str, Any]:
        """Generate a summary of the extraction results."""
        code_counts = {}
        for code in self.extracted_codes:
            code_type = code.code_type
            code_counts[code_type] = code_counts.get(code_type, 0) + 1

        return {
            'total_codes': self.total_codes_found,
            'unique_codes': len(self.get_unique_codes()),
            'codes_by_type': code_counts,
            'processing_time': self.processing_time_seconds,
            'extraction_timestamp': self.extraction_timestamp
        }

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class CodePattern(BaseModel):
    """
    Pydantic model representing a code pattern definition.
    """
    code_type: CodeType = Field(..., description="Type of medical code")
    pattern: str = Field(..., description="Regular expression pattern")
    description: str = Field(..., description="Description of the pattern")
    examples: List[str] = Field(
        default_factory=list, 
        description="Example codes that match this pattern"
    )
    priority: int = Field(
        default=1, 
        description="Priority for pattern matching (higher = more priority)"
    )
    case_sensitive: bool = Field(
        default=False, 
        description="Whether the pattern is case sensitive"
    )

    @validator('pattern')
    def validate_pattern(cls, v):
        """Validate that the regex pattern is valid."""
        import re
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f'Invalid regex pattern: {e}')
        return v

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True