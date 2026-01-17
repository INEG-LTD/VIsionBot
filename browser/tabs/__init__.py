"""
Tab Management - Multi-tab agent workflows.
"""
from .info import TabInfo
from .manager import TabManager
from .decision import TabDecisionEngine, TabDecision, TabAction

__all__ = ["TabInfo", "TabManager", "TabDecisionEngine", "TabDecision", "TabAction"]
