"""
Detects UI elements using AI vision models with numbered overlays.
"""
from typing import List, Dict, Any, Optional

from ai_utils import generate_model
from models import PageElements, PageInfo


class ElementDetector:
    """Detects UI elements using AI vision models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

