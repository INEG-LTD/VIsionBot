"""
SubAgentController - Manages sub-agents for multi-tab workflows.

Phase 3: Sub-Agent Infrastructure
"""
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Callable
import time

from .task_result import TaskResult

from .agent_context import AgentContext
from .sub_agent_result import SubAgentResult
if TYPE_CHECKING:
    from .agent_controller import AgentController
    from vision_bot import BrowserVisionBot


class SubAgentController:
    """
    Controller for managing sub-agents that work on different tabs.
    
    Responsibilities:
    - Spawn sub-agents for work on other tabs
    - Manage sub-agent lifecycle
    - Coordinate communication between parent and sub-agents
    - Track sub-agent status and results
    """
    
    def __init__(
        self,
        main_bot: "BrowserVisionBot",
        main_agent_context: AgentContext,
        controller_factory: Optional[Callable[..., "AgentController"]] = None,
        track_ineffective_actions: bool = True,
        allow_partial_completion: bool = False
    ):
        """
        Initialize SubAgentController.
        
        Args:
            main_bot: Main BrowserVisionBot instance
            main_agent_context: Context of the main agent
        """
        self.main_bot = main_bot
        self.main_agent_context = main_agent_context
        self.sub_agents: Dict[str, AgentContext] = {}  # agent_id -> AgentContext
        self.sub_agent_controllers: Dict[str, "AgentController"] = {}  # agent_id -> AgentController
        self.execution_history: List[SubAgentResult] = []
        self._pending_results: List[SubAgentResult] = []
        self._controller_factory = controller_factory
        self.track_ineffective_actions = track_ineffective_actions
        self.allow_partial_completion = allow_partial_completion
    
    def spawn_sub_agent(
        self,
        tab_id: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Spawn a sub-agent to work on a specific tab.
        
        Args:
            tab_id: Tab ID for the sub-agent to work on
            instruction: Task/instruction for the sub-agent
            metadata: Optional additional metadata
        
        Returns:
            Sub-agent ID if spawned successfully, None otherwise
        """
        if not self.main_bot.tab_manager:
            print("⚠️ Cannot spawn sub-agent: TabManager not available")
            return None
        
        # Get tab info
        tab_info = self.main_bot.tab_manager.get_tab_info(tab_id)
        if not tab_info:
            print(f"⚠️ Cannot spawn sub-agent: Tab {tab_id} not found")
            return None
        
        # Check if tab already has an agent
        existing_agent = self._get_agent_for_tab(tab_id)
        if existing_agent:
            print(f"⚠️ Tab {tab_id} already has agent {existing_agent.agent_id}")
            return existing_agent.agent_id
        
        # Create sub-agent context
        sub_agent_context = AgentContext.create_sub_agent(
            parent_agent_id=self.main_agent_context.agent_id,
            tab_id=tab_id,
            instruction=instruction,
            metadata=metadata or {}
        )
        
        # Store sub-agent
        self.sub_agents[sub_agent_context.agent_id] = sub_agent_context
        
        print(f"🤖 Spawning sub-agent: {sub_agent_context.agent_id}")
        print(f"   Tab: {tab_id} ({tab_info.purpose})")
        print(f"   Instruction: {instruction}")
        
        # Create AgentController for sub-agent
        sub_controller = self._create_agent_controller()
        
        # Store controller
        self.sub_agent_controllers[sub_agent_context.agent_id] = sub_controller
        
        # Update tab with agent ID
        self.main_bot.tab_manager.update_tab_info(
            tab_id,
            agent_id=sub_agent_context.agent_id
        )
        
        return sub_agent_context.agent_id
    
    def execute_sub_agent(self, sub_agent_id: str) -> Dict[str, Any]:
        """
        Execute a sub-agent's task.
        
        Args:
            sub_agent_id: ID of sub-agent to execute
        
        Returns:
            Result dictionary with status and result
        """
        if sub_agent_id not in self.sub_agents:
            return {
                "success": False,
                "error": f"Sub-agent {sub_agent_id} not found"
            }
        
        sub_agent_context = self.sub_agents[sub_agent_id]
        sub_controller = self.sub_agent_controllers.get(sub_agent_id)
        
        if not sub_controller:
            sub_agent_context.mark_failed("Sub-agent controller not found")
            return {
                "success": False,
                "error": "Sub-agent controller not found"
            }
        
        # Get tab info
        tab_info = self.main_bot.tab_manager.get_tab_info(sub_agent_context.tab_id)
        if not tab_info:
            sub_agent_context.mark_failed("Tab not found")
            return {
                "success": False,
                "error": "Tab not found"
            }
        
        # Switch to sub-agent's tab
        print(f"🔀 Switching to sub-agent's tab: {sub_agent_context.tab_id}")
        
        # First switch in TabManager
        if not self.main_bot.tab_manager.switch_to_tab(sub_agent_context.tab_id):
            sub_agent_context.mark_failed("Failed to switch tab in TabManager")
            return {
                "success": False,
                "error": "Failed to switch tab in TabManager"
            }
        
        # Then switch the bot's page
        self.main_bot.switch_to_page(tab_info.page)
        
        # Wait for page to be ready
        try:
            tab_info.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass  # Continue even if timeout
        
        # Verify we're on the correct page - check both the page and bot's components
        try:
            page_url = tab_info.page.url
            bot_url = self.main_bot.page.url if getattr(self.main_bot, "page", None) else "no page"
            session_tracker_page = getattr(self.main_bot.session_tracker, "page", None) if getattr(self.main_bot, "session_tracker", None) else None
            session_tracker_url = session_tracker_page.url if session_tracker_page else "no page"
            
            print(f"   ✅ Switched to tab")
            print(f"      Page URL: {page_url}")
            print(f"      Bot.page URL: {bot_url}")
            print(f"      SessionTracker.page URL: {session_tracker_url}")
        except Exception as e:
            print(f"   ⚠️ Error verifying page switch: {e}")
        
        # Mark as running
        sub_agent_context.mark_running()
        
        try:
            # Execute sub-agent's task
            print(f"▶️ Executing sub-agent {sub_agent_id}: {sub_agent_context.instruction}")
            start_time = time.time()
            goal_result = sub_controller.run_agentic_mode(
                sub_agent_context.instruction,
                agent_context=sub_agent_context
            )
            end_time = time.time()
            
            # Mark as completed
            success = goal_result.success if isinstance(goal_result, TaskResult) else getattr(goal_result, 'success', False)
            status_str = 'achieved' if success else 'failed'
            # Handle both TaskResult and legacy GoalResult for compatibility
            if hasattr(goal_result, 'status') and isinstance(goal_result.status, str):
                status_str = goal_result.status
            elif hasattr(goal_result, 'status') and hasattr(goal_result.status, 'value'):
                status_str = goal_result.status.value
            result = SubAgentResult(
                agent_id=sub_agent_id,
                tab_id=sub_agent_context.tab_id,
                instruction=sub_agent_context.instruction,
                success=success,
                status=status_str,
                confidence=getattr(goal_result, 'confidence', 0.0),
                reasoning=getattr(goal_result, 'reasoning', ''),
                evidence=getattr(goal_result, 'evidence', None) or {},
                error=None if success else getattr(goal_result, 'reasoning', ''),
                started_at=start_time,
                completed_at=end_time,
                metadata=sub_agent_context.metadata.copy()
            )
            self._record_result(sub_agent_context, result)
            
            print(f"✅ Sub-agent {sub_agent_id} completed")
            print(f"   Status: {result.status}")
            print(f"   Confidence: {result.confidence:.2f}")
            
            return result.to_dict()
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Sub-agent {sub_agent_id} failed: {error_msg}")
            end_time = time.time()
            failure_result = SubAgentResult(
                agent_id=sub_agent_id,
                tab_id=sub_agent_context.tab_id,
                instruction=sub_agent_context.instruction,
                success=False,
                status='failed',
                confidence=0.0,
                reasoning=f"Sub-agent execution failed: {error_msg}",
                evidence={},
                error=error_msg,
                started_at=start_time,
                completed_at=end_time,
                metadata=sub_agent_context.metadata.copy()
            )
            self._record_result(sub_agent_context, failure_result, mark_failed=True)
            return failure_result.to_dict()
    
    def get_sub_agent(self, sub_agent_id: str) -> Optional[AgentContext]:
        """Get sub-agent context by ID"""
        return self.sub_agents.get(sub_agent_id)
    
    def list_sub_agents(self, status: Optional[str] = None) -> List[AgentContext]:
        """
        List all sub-agents, optionally filtered by status.
        
        Args:
            status: Optional status to filter by (pending, running, completed, failed)
        
        Returns:
            List of AgentContext objects
        """
        if status is None:
            return list(self.sub_agents.values())
        return [agent for agent in self.sub_agents.values() if agent.status == status]
    
    def get_sub_agent_result(self, sub_agent_id: str) -> Optional[Dict[str, Any]]:
        """Get result from a completed sub-agent"""
        sub_agent = self.sub_agents.get(sub_agent_id)
        if sub_agent and sub_agent.status == "completed":
            return sub_agent.result
        return None
    
    def _get_agent_for_tab(self, tab_id: str) -> Optional[AgentContext]:
        """Get agent context for a specific tab"""
        for agent in self.sub_agents.values():
            if agent.tab_id == tab_id:
                return agent
        return None
    
    def cleanup_completed_agents(self) -> int:
        """
        Clean up completed sub-agents.
        
        Returns:
            Number of agents cleaned up
        """
        completed = [agent_id for agent_id, agent in self.sub_agents.items() if agent.status == "completed"]
        
        for agent_id in completed:
            # Remove controller
            if agent_id in self.sub_agent_controllers:
                del self.sub_agent_controllers[agent_id]
            
            # Remove agent context
            del self.sub_agents[agent_id]
        
        if completed:
            print(f"🧹 Cleaned up {len(completed)} completed sub-agent(s)")
        
        return len(completed)

    def pop_completed_results(self) -> List[SubAgentResult]:
        """
        Retrieve newly completed sub-agent results.
        """
        results = self._pending_results[:]
        self._pending_results = []
        return results

    def get_execution_history(self) -> List[SubAgentResult]:
        """Return the full execution history."""
        return list(self.execution_history)

    def _record_result(
        self,
        sub_agent_context: AgentContext,
        result: SubAgentResult,
        mark_failed: bool = False
    ) -> None:
        if mark_failed:
            sub_agent_context.mark_failed(result.error)
        else:
            sub_agent_context.mark_completed(result.to_dict())
        self.execution_history.append(result)
        self._pending_results.append(result)

    def _create_agent_controller(self) -> "AgentController":
        base_knowledge = self._get_base_knowledge()
        parallel_completion_and_action = getattr(self.main_bot, "parallel_completion_and_action", True)
        if self._controller_factory:
            try:
                return self._controller_factory(
                    self.main_bot,
                    base_knowledge=base_knowledge,
                    track_ineffective_actions=self.track_ineffective_actions,
                    allow_partial_completion=self.allow_partial_completion,
                    parallel_completion_and_action=parallel_completion_and_action
                )
            except TypeError:
                try:
                    return self._controller_factory(
                        self.main_bot,
                        base_knowledge=base_knowledge,
                        track_ineffective_actions=self.track_ineffective_actions,
                        allow_partial_completion=self.allow_partial_completion
                    )
                except TypeError:
                    try:
                        return self._controller_factory(self.main_bot, base_knowledge=base_knowledge)
                    except TypeError:
                        return self._controller_factory(self.main_bot)
        # Import here to avoid circular import
        from .agent_controller import AgentController
        return AgentController(
            bot=self.main_bot,
            track_ineffective_actions=self.track_ineffective_actions,
            base_knowledge=base_knowledge,
            allow_partial_completion=self.allow_partial_completion,
            parallel_completion_and_action=parallel_completion_and_action
        )

    def _get_base_knowledge(self) -> Optional[List[str]]:
        if hasattr(self.main_bot, "session_tracker"):
            base = getattr(self.main_bot.session_tracker, "base_knowledge", None)
            if base:
                base_list = base.copy()
            else:
                base_list = []
        else:
            base_list = []
        helper_rule = (
            "You are a delegated sub-agent helper. Stay focused on the provided instruction, "
            "do not spawn additional sub-agents, and do not initiate parallel orchestration unless explicitly commanded."
        )
        if helper_rule not in base_list:
            base_list.append(helper_rule)
        return base_list

