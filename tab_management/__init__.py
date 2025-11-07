"""
Tab Management - Phase 1 & 2: Foundation and Decision Making

Provides tab lifecycle management and LLM-based decision making for multi-tab agent workflows.
"""
from .tab_info import TabInfo
from .tab_manager import TabManager
from .tab_decision_engine import TabDecisionEngine, TabDecision, TabAction

__all__ = ["TabInfo", "TabManager", "TabDecisionEngine", "TabDecision", "TabAction"]

