"""
ActionResult - Structured return type for act() and extract() operations.

Provides consistent return type with success status, data, and metadata.
"""
from dataclasses import dataclass, field
from typing import Any, Optional, Dict


@dataclass
class ActionResult:
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


