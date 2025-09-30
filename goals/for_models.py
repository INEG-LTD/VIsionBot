"""
Shared models for ForGoal and TargetResolver to avoid circular imports.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class IterationTargetsResponse(BaseModel):
    """Response model for vision-based target detection"""
    targets: List[Dict[str, Any]] = Field(description="List of detected iteration targets")
    reasoning: str = Field(description="Explanation of target detection")
    confidence: float = Field(description="Confidence in target detection (0-1)")


class ElementContextResponse(BaseModel):
    """Response model for element context extraction"""
    context: Dict[str, Any] = Field(description="Rich context information for the element")
    actionability: List[str] = Field(description="Available actions for this element")
    visual_description: str = Field(description="Visual description of the element")


class TargetValidationResponse(BaseModel):
    """Response model for target validation"""
    validations: List[bool] = Field(description="Validation result for each target")
    reasoning: str = Field(description="Explanation of validation results")
