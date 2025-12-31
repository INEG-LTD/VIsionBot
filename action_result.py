"""
ActionResult - Structured return type for act() and extract() operations.

Provides consistent return type with success status, data, and metadata.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, Dict


@dataclass
class ActionResult:
    """
    Structured result for act() and extract() operations.
    
    Provides consistent return type with success status, data, and metadata.
    Similar to AgentResult but for individual actions and extractions.
    
    Attributes:
        success: Whether the action/extraction succeeded
        message: Human-readable message describing the result
        data: The actual data (for extract) or action context (for act)
        confidence: Confidence score (0.0-1.0)
        metadata: Additional metadata about the operation
        error: Error message if operation failed (None if successful)
    
    Example:
        >>> result = bot.act("Click button", return_result=True)
        >>> if result.success:
        ...     print(f"Success: {result.message}")
        ...     print(f"Confidence: {result.confidence}")
        
        >>> result = bot.extract("Get price", return_result=True)
        >>> if result.success:
        ...     price = result.data  # The extracted data
        ...     print(f"Confidence: {result.confidence}")
    """
    success: bool
    message: str = ""
    data: Optional[Any] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def __bool__(self) -> bool:
        """
        Allow truthiness check.
        
        Enables: if result: ...
        
        Returns:
            bool: True if success, False otherwise
        """
        return self.success
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        status = "✅" if self.success else "❌"
        data_info = f", data={type(self.data).__name__}" if self.data is not None else ""
        return f"ActionResult({status}, message='{self.message}', confidence={self.confidence:.2f}{data_info})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict containing all result fields
        """
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "error": self.error,
        }


