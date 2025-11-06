"""
Extract Goal - Validates data extraction from web pages.
"""
from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel, Field

from .base import BaseGoal, GoalResult, GoalStatus, GoalContext, EvaluationTiming


class ExtractGoal(BaseGoal):
    """
    Goal for extracting data from web pages.
    
    Validates that the requested data was successfully extracted
    and matches the expected format/structure.
    """
    
    EVALUATION_TIMING = EvaluationTiming.AFTER  # Extract after page load/interaction
    
    def __init__(
        self,
        description: str,
        extraction_prompt: str,  # What to extract (e.g., "product price")
        output_format: str = "json",  # "json", "text", "structured"
        model_schema: Optional[Type[BaseModel]] = None,
        expected_fields: Optional[List[str]] = None,  # For validation
        scope: str = "viewport",  # "viewport", "full_page", "element"
        element_description: Optional[str] = None,
        **kwargs
    ):
        super().__init__(description, **kwargs)
        self.extraction_prompt = extraction_prompt
        self.output_format = output_format
        self.model_schema = model_schema
        self.expected_fields = expected_fields or []
        self.scope = scope
        self.element_description = element_description
        
        # Extraction state
        self.extraction_attempted = False
        self.extraction_result: Optional[Union[str, Dict, BaseModel]] = None
        self.extraction_confidence: float = 0.0
        self.extraction_error: Optional[str] = None
        self.bot_reference: Optional[Any] = None  # Reference to bot for extract method
    
    def set_bot_reference(self, bot) -> None:
        """Set reference to bot instance for extract method access"""
        self.bot_reference = bot
    
    def _get_bot_reference(self, context: GoalContext) -> Optional[Any]:
        """Get bot reference from context or stored reference"""
        if self.bot_reference:
            return self.bot_reference
        
        # Try to get from context
        if hasattr(context, 'bot_reference'):
            return context.bot_reference
        
        # Try to get from goal monitor
        if hasattr(self, 'goal_monitor') and hasattr(self.goal_monitor, 'bot_reference'):
            return self.goal_monitor.bot_reference
        
        return None
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """
        Evaluate if extraction was successful.
        
        Flow:
        1. Check if extraction has been performed
        2. Validate the extracted data
        3. Return appropriate status
        """
        page = context.page_reference
        if not page:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.0,
                reasoning="No page reference available for extraction"
            )
        
        # Check if extraction already happened
        if not self.extraction_attempted:
            # Try to perform extraction
            return self._perform_extraction(context)
        
        # Validate extraction result
        if self.extraction_error:
            # Request retry if we haven't exceeded max retries
            if self.can_retry():
                retry_reason = f"Extraction failed: {self.extraction_error}"
                if self.request_retry(retry_reason):
                    # Reset state for retry
                    self.extraction_attempted = False
                    self.extraction_error = None
                    return GoalResult(
                        status=GoalStatus.PENDING,
                        confidence=0.3,
                        reasoning=f"Extraction failed, requesting retry: {self.extraction_error}",
                        next_actions=["Retry extraction with different strategy"]
                    )
            
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning=f"Extraction failed: {self.extraction_error}",
                evidence={"error": self.extraction_error}
            )
        
        # Validate extracted data structure
        validation_result = self._validate_extraction()
        if validation_result["is_valid"]:
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=self.extraction_confidence,
                reasoning=f"Successfully extracted: {self.extraction_prompt}",
                evidence={
                    "extraction_result": self._serialize_result(),
                    "format": self.output_format,
                    "validation": validation_result
                }
            )
        else:
            # Request retry if validation fails
            if self.can_retry():
                retry_reason = f"Extracted data doesn't match expected format: {validation_result['reason']}"
                if self.request_retry(retry_reason):
                    # Reset for retry
                    self.extraction_attempted = False
                    return GoalResult(
                        status=GoalStatus.PENDING,
                        confidence=0.5,
                        reasoning=f"Extraction validation failed, requesting retry: {validation_result['reason']}",
                        next_actions=["Retry extraction with different strategy"]
                    )
            
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.5,
                reasoning=f"Extraction validation failed: {validation_result['reason']}",
                evidence={"validation_error": validation_result['reason']}
            )
    
    def _perform_extraction(self, context: GoalContext) -> GoalResult:
        """Perform the actual extraction using the bot's extract method"""
        try:
            # Get reference to bot's extract method
            bot = self._get_bot_reference(context)
            
            if not bot or not hasattr(bot, 'extract'):
                return GoalResult(
                    status=GoalStatus.UNKNOWN,
                    confidence=0.0,
                    reasoning="Bot extract method not available"
                )
            
            # Perform extraction
            self.extraction_attempted = True
            print(f"[ExtractGoal] Performing extraction: {self.extraction_prompt}")
            
            result = bot.extract(
                prompt=self.extraction_prompt,
                output_format=self.output_format,
                model_schema=self.model_schema,
                scope=self.scope,
                element_description=self.element_description
            )
            
            self.extraction_result = result
            
            # Extract confidence if available
            if isinstance(result, dict) and "confidence" in result:
                self.extraction_confidence = result.pop("confidence", 0.0)
            elif isinstance(result, dict) and "_confidence" in result:
                self.extraction_confidence = result.pop("_confidence", 0.0)
            else:
                self.extraction_confidence = 0.8  # Default confidence
            
            # Return pending to trigger validation in next evaluation
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=self.extraction_confidence,
                reasoning="Extraction performed, validating result..."
            )
            
        except Exception as e:
            self.extraction_attempted = True
            self.extraction_error = str(e)
            import traceback
            traceback.print_exc()
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning=f"Extraction error: {e}",
                evidence={"error": str(e)}
            )
    
    def _validate_extraction(self) -> Dict[str, Any]:
        """Validate that extracted data matches expectations"""
        
        if self.extraction_result is None:
            return {
                "is_valid": False,
                "reason": "No extraction result"
            }
        
        # Validate based on output format
        if self.output_format == "json":
            if not isinstance(self.extraction_result, dict):
                return {
                    "is_valid": False,
                    "reason": f"Expected dict, got {type(self.extraction_result).__name__}"
                }
            
            # Check expected fields if specified
            if self.expected_fields:
                missing = [field for field in self.expected_fields 
                          if field not in self.extraction_result]
                if missing:
                    return {
                        "is_valid": False,
                        "reason": f"Missing expected fields: {missing}"
                    }
        
        elif self.output_format == "text":
            if not isinstance(self.extraction_result, str):
                return {
                    "is_valid": False,
                    "reason": f"Expected string, got {type(self.extraction_result).__name__}"
                }
            if not self.extraction_result.strip():
                return {
                    "is_valid": False,
                    "reason": "Extracted text is empty"
                }
        
        elif self.output_format == "structured":
            if self.model_schema and not isinstance(self.extraction_result, self.model_schema):
                return {
                    "is_valid": False,
                    "reason": f"Result doesn't match expected schema {self.model_schema.__name__}"
                }
        
        return {
            "is_valid": True,
            "reason": "Extraction validated successfully"
        }
    
    def _serialize_result(self) -> Union[str, Dict[str, Any]]:
        """Serialize extraction result for evidence"""
        if isinstance(self.extraction_result, BaseModel):
            return self.extraction_result.model_dump()
        elif isinstance(self.extraction_result, dict):
            return self.extraction_result
        else:
            return str(self.extraction_result)
    
    def get_description(self, context: GoalContext) -> str:
        """Generate a detailed description of what this goal is looking for"""
        status = "PENDING"
        if self.extraction_attempted:
            if self.extraction_error:
                status = "FAILED"
            elif self.extraction_result:
                validation = self._validate_extraction()
                status = "ACHIEVED" if validation["is_valid"] else "VALIDATION_FAILED"
        
        result_preview = ""
        if self.extraction_result:
            if isinstance(self.extraction_result, str):
                result_preview = self.extraction_result[:100]
            elif isinstance(self.extraction_result, dict):
                result_preview = str(list(self.extraction_result.keys()))[:100]
            else:
                result_preview = str(type(self.extraction_result).__name__)
        
        return f"""
Goal Type: Extract Goal
Description: {self.description}
Extraction Prompt: {self.extraction_prompt}
Output Format: {self.output_format}
Scope: {self.scope}
Current Status: {status}
Expected Fields: {', '.join(self.expected_fields) if self.expected_fields else 'None'}
Progress: {'✅ Extraction completed and validated' if status == 'ACHIEVED' else '⏳ Waiting for extraction' if status == 'PENDING' else '❌ Extraction failed'}
Result Preview: {result_preview if result_preview else 'None'}
Issues: {'None' if status == 'ACHIEVED' else self.extraction_error if self.extraction_error else 'Extraction not yet performed'}
"""


