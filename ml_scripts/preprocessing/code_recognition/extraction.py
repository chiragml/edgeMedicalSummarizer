"""
Medical Code Extraction Logic

This module contains the main logic for extracting medical billing codes from text
using regular expressions and pattern matching.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime

from .models import MedicalCode, ExtractionResult, CodeType, CodePattern

logger = logging.getLogger(__name__)


class CodeExtractor:
    """
    Main class for extracting medical billing codes from text.
    """
    
    def __init__(self, patterns_file: Optional[str] = None):
        """
        Initialize the CodeExtractor.
        
        Args:
            patterns_file: Path to the JSON file containing code patterns
        """
        self.patterns_file = patterns_file or self._get_default_patterns_file()
        self.patterns: Dict[str, List[CodePattern]] = {}
        self.compiled_patterns: Dict[str, List[Tuple[re.Pattern, CodePattern]]] = {}
        
        self._load_patterns()
        self._compile_patterns()
    
    def _get_default_patterns_file(self) -> str:
        """Get the default patterns file path."""
        current_dir = Path(__file__).parent
        return str(current_dir / "code_patterns.json")
    
    def _load_patterns(self) -> None:
        """Load patterns from the JSON file."""
        try:
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                patterns_data = json.load(f)
            
            self.patterns = {}
            for code_type_str, pattern_list in patterns_data.items():
                try:
                    code_type = CodeType(code_type_str)
                    self.patterns[code_type_str] = []
                    
                    for pattern_data in pattern_list:
                        pattern = CodePattern(
                            code_type=code_type,
                            pattern=pattern_data['pattern'],
                            description=pattern_data['description'],
                            examples=pattern_data.get('examples', []),
                            priority=pattern_data.get('priority', 1),
                            case_sensitive=pattern_data.get('case_sensitive', False)
                        )
                        self.patterns[code_type_str].append(pattern)
                
                except ValueError as e:
                    logger.warning(f"Skipping invalid code type '{code_type_str}': {e}")
                    continue
            
            logger.info(f"Loaded {len(self.patterns)} code types with patterns")
            
        except FileNotFoundError:
            logger.error(f"Patterns file not found: {self.patterns_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in patterns file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            raise
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for better performance."""
        self.compiled_patterns = {}
        
        for code_type_str, pattern_list in self.patterns.items():
            self.compiled_patterns[code_type_str] = []
            
            for pattern in pattern_list:
                try:
                    flags = 0 if pattern.case_sensitive else re.IGNORECASE
                    compiled_pattern = re.compile(pattern.pattern, flags)
                    self.compiled_patterns[code_type_str].append((compiled_pattern, pattern))
                except re.error as e:
                    logger.warning(f"Failed to compile pattern for {code_type_str}: {e}")
                    continue
    
    def extract_codes(
        self, 
        text: str, 
        code_types: Optional[List[CodeType]] = None,
        include_context: bool = True,
        context_window: int = 50
    ) -> ExtractionResult:
        """
        Extract medical codes from the given text.
        
        Args:
            text: Input text to extract codes from
            code_types: Specific code types to extract (None for all)
            include_context: Whether to include surrounding context
            context_window: Number of characters around the code for context
            
        Returns:
            ExtractionResult containing all extracted codes
        """
        start_time = time.time()
        extracted_codes = []
        
        # Filter code types if specified
        target_types = []
        if code_types:
            target_types = [ct.value for ct in code_types]
        else:
            target_types = list(self.compiled_patterns.keys())
        
        # Sort patterns by priority (higher priority first)
        sorted_patterns = []
        for code_type_str in target_types:
            if code_type_str in self.compiled_patterns:
                patterns = self.compiled_patterns[code_type_str]
                # Sort by priority (descending)
                patterns = sorted(patterns, key=lambda x: x[1].priority, reverse=True)
                sorted_patterns.extend([(code_type_str, p[0], p[1]) for p in patterns])
        
        # Sort all patterns by priority
        sorted_patterns.sort(key=lambda x: x[2].priority, reverse=True)
        
        # Track already extracted positions to avoid overlaps
        extracted_positions = set()
        
        # Extract codes using each pattern
        for code_type_str, compiled_pattern, pattern in sorted_patterns:
            matches = compiled_pattern.finditer(text)
            
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                
                # Check for overlaps with already extracted codes
                if any(pos in range(start_pos, end_pos) for pos in extracted_positions):
                    continue
                
                # Extract the matched code
                matched_text = match.group()
                
                # For patterns with groups, extract the actual code from the first group
                if match.groups() and match.group(1):
                    code_value = match.group(1)
                else:
                    code_value = matched_text
                
                # Get context if requested
                context = None
                if include_context:
                    context_start = max(0, start_pos - context_window)
                    context_end = min(len(text), end_pos + context_window)
                    context = text[context_start:context_end].strip()
                
                # Create MedicalCode object
                medical_code = MedicalCode(
                    code_type=CodeType(code_type_str),
                    code_value=code_value,
                    original_text=matched_text,
                    start_position=start_pos,
                    end_position=end_pos,
                    context=context,
                    metadata={
                        'pattern_description': pattern.description,
                        'pattern_priority': pattern.priority,
                        'match_confidence': self._calculate_confidence(code_value, pattern)
                    }
                )
                
                extracted_codes.append(medical_code)
                
                # Mark positions as extracted
                for pos in range(start_pos, end_pos):
                    extracted_positions.add(pos)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create and return result
        result = ExtractionResult(
            input_text=text,
            extracted_codes=extracted_codes,
            extraction_timestamp=datetime.now().isoformat(),
            processing_time_seconds=round(processing_time, 4),
            extraction_metadata={
                'patterns_used': len(sorted_patterns),
                'code_types_searched': target_types,
                'include_context': include_context,
                'context_window': context_window
            }
        )
        
        return result
    
    def _calculate_confidence(self, code_value: str, pattern: CodePattern) -> float:
        """
        Calculate confidence score for an extracted code.
        
        Args:
            code_value: The extracted code value
            pattern: The pattern that matched the code
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Higher confidence for higher priority patterns
        confidence += (pattern.priority - 1) * 0.1
        
        # Higher confidence if code matches examples exactly
        if code_value.upper() in [ex.upper() for ex in pattern.examples]:
            confidence += 0.3
        
        # Higher confidence for longer codes (more specific)
        if len(code_value) >= 5:
            confidence += 0.1
        
        # Ensure confidence is between 0.0 and 1.0
        return min(1.0, max(0.0, confidence))
    
    def extract_codes_by_type(
        self, 
        text: str, 
        code_type: CodeType,
        include_context: bool = True
    ) -> List[MedicalCode]:
        """
        Extract codes of a specific type only.
        
        Args:
            text: Input text to extract codes from
            code_type: Specific code type to extract
            include_context: Whether to include surrounding context
            
        Returns:
            List of extracted MedicalCode objects
        """
        result = self.extract_codes(text, [code_type], include_context)
        return result.extracted_codes
    
    def validate_code(self, code_value: str, code_type: CodeType) -> bool:
        """
        Validate if a code matches the patterns for its type.
        
        Args:
            code_value: The code value to validate
            code_type: The expected code type
            
        Returns:
            True if the code is valid for the given type
        """
        code_type_str = code_type.value
        
        if code_type_str not in self.compiled_patterns:
            return False
        
        for compiled_pattern, _ in self.compiled_patterns[code_type_str]:
            if compiled_pattern.fullmatch(code_value):
                return True
        
        return False
    
    def get_pattern_info(self, code_type: CodeType) -> List[Dict[str, Any]]:
        """
        Get information about patterns for a specific code type.
        
        Args:
            code_type: The code type to get pattern info for
            
        Returns:
            List of pattern information dictionaries
        """
        code_type_str = code_type.value
        
        if code_type_str not in self.patterns:
            return []
        
        pattern_info = []
        for pattern in self.patterns[code_type_str]:
            pattern_info.append({
                'pattern': pattern.pattern,
                'description': pattern.description,
                'examples': pattern.examples,
                'priority': pattern.priority,
                'case_sensitive': pattern.case_sensitive
            })
        
        return pattern_info
    
    def get_supported_code_types(self) -> List[CodeType]:
        """
        Get list of supported code types.
        
        Returns:
            List of supported CodeType enums
        """
        return [CodeType(code_type_str) for code_type_str in self.patterns.keys()]


# Convenience functions
def extract_medical_codes(
    text: str, 
    patterns_file: Optional[str] = None,
    code_types: Optional[List[CodeType]] = None
) -> ExtractionResult:
    """
    Convenience function to extract medical codes from text.
    
    Args:
        text: Input text to extract codes from
        patterns_file: Optional path to patterns file
        code_types: Specific code types to extract
        
    Returns:
        ExtractionResult containing extracted codes
    """
    extractor = CodeExtractor(patterns_file)
    return extractor.extract_codes(text, code_types)


def quick_extract_codes(text: str) -> List[str]:
    """
    Quick extraction that returns just the code values as strings.
    
    Args:
        text: Input text to extract codes from
        
    Returns:
        List of extracted code values
    """
    result = extract_medical_codes(text)
    return [code.code_value for code in result.extracted_codes]