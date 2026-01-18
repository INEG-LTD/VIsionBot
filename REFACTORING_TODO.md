# Refactoring TODO

## Completed ✅

### core/executor.py → core/executor/ (DONE - Committed)
Split 1,736-line file into 4 focused modules:
- **base.py** (426 lines) - Core Executor initialization and plan execution
- **callbacks.py** (185 lines) - Pre/post-action callback system
- **actions.py** (1,062 lines) - All action type implementations
- **ui_feedback.py** (164 lines) - Visual feedback and highlighting
- **__init__.py** (49 lines) - Module exports and method binding

**Status**: ✅ Committed (c545041)

---

## Remaining Work 📋

### 1. agent/agent_controller.py → agent/agent/ (ANALYZED - Ready to implement)
**Size**: 4,097 lines
**Complexity**: HIGH (main execution loop is 866 lines alone)

**Recommended Split (7 modules)**:

#### agent/agent/base.py (~400 lines)
**Purpose**: Core initialization, pause/resume control, mini goals
**Methods**:
- `__init__` (lines 118-318)
- `pause`, `resume`, `is_paused`, `_check_pause` (lines 369-469)
- `register_mini_goal`, `_handle_mini_goal_trigger` (lines 319-355)
- `_get_current_prompt`, `_is_nav_action` (lines 356-368)

#### agent/agent/execution_loop.py (~900 lines)
**Purpose**: Main task execution loop
**Methods**:
- `run_execute_task` (lines 470-1336) - The massive 866-line main loop
- `_capture_snapshot` (lines 1336-1397)

**Notes**: This single method is too large and should be further refactored, but keeping it as one module for now.

#### agent/agent/action_planning.py (~350 lines)
**Purpose**: Action plan management and element resolution
**Methods**:
- `_determine_next_action` (lines 1398-1431)
- `_start_action_plan`, `_clear_action_plan` (lines 1432-1449)
- `_peek_pending_action_plan_step`, `_pop_pending_action_plan_step` (lines 1450-1457)
- `_resolve_overlay_for_step` (lines 1458-1485)
- `_prepare_plan_step_action` (lines 1486-1503)
- `_is_overlay_clipped`, `_build_scroll_action_for_overlay` (lines 1504-1529)
- `_log_pre_generated_plan_step` (lines 1530-1562)
- `_find_overlay_by_index`, `_find_overlay_index_by_description`, `_find_overlay_index_by_metadata` (lines 1563-1586)
- `_get_page_info_dimension` (lines 1587-1597)
- `_url_matches_target`, `_history_steps_to_target` (lines 1598-1637)
- `_filter_navigation_commands` (lines 1638-1653)

#### agent/agent/extraction.py (~650 lines)
**Purpose**: Extraction detection and coordination
**Methods**:
- `_handle_internal_command` (lines 1654-1697)
- `_detect_extraction_need` (lines 1747-1825)
- `_detect_url_extraction_need`, `_extract_url_from_element` (lines 1826-1918)
- `_normalize_extraction_prompt`, `_normalize_task_name` (lines 1919-1928)
- `_initialize_requirement_flags`, `_update_requirement_flags_from_text` (lines 1929-1950)
- `_infer_extraction_subject`, `_build_extraction_task_description` (lines 1951-1972)
- `_build_comprehensive_extraction_prompt` (lines 1973-2001)
- `_extract_requested_subjects` (lines 2002-2031)
- `_build_manual_parallel_plan` (lines 2032-2093)
- `_ensure_primary_output_tasks`, `_activate_primary_output_task` (lines 2212-2255)
- `_should_skip_extraction`, `_record_extraction_failure`, `_record_extraction_success` (lines 2458-2470)
- `_retarget_after_extraction_failure` (lines 2471-2527)

#### agent/agent/action_parsing.py (~400 lines)
**Purpose**: Action parsing and task management utilities
**Methods**:
- `_parse_action_for_act_params` (lines 2528-2629)
- `_determine_scroll_direction_from_position` (lines 2630-2840)
- `_handle_defer_input` (lines 2094-2126)
- `_handle_ask_command` (lines 2127-2211)
- `_register_task`, `_mark_task_completed`, `_mark_task_failed` (lines 2256-2293)
- `_record_action_outcome`, `_update_task_blockers` (lines 2294-2368)
- `_build_current_task_prompt` (lines 2369-2457)

#### agent/agent/completion.py (~500 lines)
**Purpose**: Completion detection and state checking
**Methods**:
- `_is_in_exploration_mode` (lines 2841-2874)
- `_simple_completion_check` (lines 2875-2937)
- `_get_page_state` (lines 2938-2983)
- `_get_ui_state_info` (lines 2984-3037)
- `_build_evidence` (lines 3087-3133)
- `_rewrite_task_prompt_using_completion_reasoning` (lines 3134-3205)
- `_build_tab_summary` (lines 3206-3221)
- `_page_state_changed` (lines 3222-3257)
- `_detect_ui_state_changes` (lines 3258-3312)

#### agent/agent/subagent.py (~650 lines)
**Purpose**: Sub-agent orchestration and parallelization
**Methods**:
- `_update_sub_agent_policy` (lines 3313-3351)
- `_compute_sub_agent_policy` (lines 3352-3459)
- `_query_sub_agent_policy_yes_no` (lines 3460-3487)
- `_build_sub_agent_policy_yes_no_system_prompt`, `_build_sub_agent_policy_yes_no_prompt` (lines 3488-3535)
- `_query_sub_agent_policy_llm` (lines 3536-3562)
- `_build_sub_agent_policy_system_prompt`, `_build_sub_agent_policy_prompt` (lines 3563-3633)
- `_normalize_suggested_url` (lines 3634-3643)
- `_run_orchestrated_task` (lines 3644-3743)
- `_orchestrate_parallel_work` (lines 3744-3793)
- `_create_parallel_plan` (lines 3794-3814)
- `_build_parallel_plan_system_prompt`, `_build_parallel_plan_prompt` (lines 3815-3858)
- `_can_spawn_sub_agent` (lines 3859-3873)
- `_spawn_child_controller`, `_drain_sub_agent_results` (lines 3038-3086)
- `get_sub_agent_results` (lines 4086-4089)
- `_apply_sub_agent_override`, `_policy_display_name` (lines 1698-1746)

#### agent/agent/utilities.py (~250 lines)
**Purpose**: Tab decisions, logging, and misc helpers
**Methods**:
- `_execute_tab_decision` (lines 3874-4085)
- `_log_event` (lines 4090-4097)

#### agent/agent/__init__.py
**Purpose**: Module exports and mixin composition
**Pattern**: Use mixins to compose the Agent class:
```python
from .base import AgentBaseMixin
from .execution_loop import AgentExecutionLoopMixin
from .action_planning import AgentActionPlanningMixin
from .extraction import AgentExtractionMixin
from .action_parsing import AgentActionParsingMixin
from .completion import AgentCompletionMixin
from .subagent import AgentSubAgentMixin
from .utilities import AgentUtilitiesMixin

class Agent(
    AgentBaseMixin,
    AgentExecutionLoopMixin,
    AgentActionPlanningMixin,
    AgentExtractionMixin,
    AgentActionParsingMixin,
    AgentCompletionMixin,
    AgentSubAgentMixin,
    AgentUtilitiesMixin,
    TaskBasedExecutionMixin,  # Existing mixin
):
    """Agent controller with modular implementation"""
    pass

__all__ = ["Agent", "UserQuestionCallback"]
```

**Complexity Notes**:
- `run_execute_task` (866 lines) is too large even for its own module - should be refactored further
- Many interdependencies between extraction, completion, and sub-agent modules
- State management is scattered across multiple attributes
- Consider creating a `state.py` module for centralized state management in future refactoring

---

### 2. core/browser.py → core/browser/ (ANALYZED - Not started)
**Size**: 3,955 lines
**Complexity**: VERY HIGH (most complex file in codebase)

**Recommended Split (7 modules)**:

#### core/browser/core.py (~550 lines)
**Purpose**: Initialization, lifecycle management, configuration
**Methods**: `__init__`, `init_browser`, `start`, `end`, `__enter__`, `__exit__`, `_check_termination`, `use`

#### core/browser/page_manager.py (~350 lines)
**Purpose**: Page/tab management, auto-on-load actions
**Methods**: `switch_to_page`, `on_new_page_load`, `_attach_page_load_handler`, `_run_auto_actions_for_current_page`, etc.

#### core/browser/action_executor.py (~450 lines)
**Purpose**: Main `act()` method and action routing
**Methods**: `act`, `_execute_keyword_command`, `_record_history_entry`

#### core/browser/keyword_handlers.py (~750 lines)
**Purpose**: All keyword command implementations
**Methods**: `_keyword_overlay_action`, `_keyword_click`, `_keyword_type`, `_keyword_select`, `_keyword_upload`, `_keyword_datetime`, `_keyword_scroll`, `_keyword_wait`, `_keyword_press`, `_keyword_stop`, `_keyword_defer`, `_keyword_back`, `_keyword_forward`, `_keyword_open`, etc.

#### core/browser/extraction.py (~750 lines)
**Purpose**: Data extraction functionality
**Methods**: `extract`, `extract_batch`, `extract_multi_field`, `_enrich_extracted_data_with_urls`

#### core/browser/agent.py (~450 lines)
**Purpose**: Agent execution and control
**Methods**: `execute_mission`, `pause_agent`, `resume_agent`, `is_agent_paused`

#### core/browser/helpers.py (~600 lines)
**Purpose**: Utilities, caching, queue, convenience methods
**Methods**: Plan caching, defer input, navigation (`goto`), multi-command, mini goals, semantic/DOM helpers, parsing, command history, queue, convenience (`get_url`, `get_title`, `wait_for_load`, `screenshot`), properties

**Complexity Notes**:
- Very high interdependency between modules
- Each keyword handler needs access to executor, page, session_tracker
- Extraction needs access to many bot internals
- Consider mixin pattern similar to agent split
- May benefit from extracting a `BrowserState` class to centralize state management

---

## Implementation Strategy

### For agent/agent_controller.py:

1. **Create module stubs** - Create all 7 files with imports and class definitions
2. **Extract methods bottom-up** - Start with least dependent (utilities, completion), then extraction, action_planning, subagent, execution_loop, and finally base
3. **Use mixin pattern** - Each module is a mixin that Agent inherits from
4. **Test incrementally** - Test imports and basic functionality after each module
5. **Update imports** - Find and update all imports in codebase

### For core/browser.py:

1. **Similar approach to agent**
2. **Higher complexity** - More careful dependency management needed
3. **Consider state extraction** - May benefit from extracting state management first
4. **Incremental testing** - Critical due to complexity

---

## Testing Strategy

After each split:
1. ✅ Test imports: `from core.executor import Executor` (done)
2. ⏳ Test imports: `from agent import Agent`
3. ⏳ Test imports: `from core.browser import Browser`
4. ⏳ Run unit tests if they exist
5. ⏳ Run integration test (demo.py)
6. ⏳ Commit changes with detailed message

---

## Current Status

**Completed**:
- ✅ core/executor.py split (committed)
- ✅ Analysis of agent_controller.py structure
- ✅ Analysis of browser.py structure (from previous session)

**Next Steps**:
1. Implement agent/agent_controller.py split
2. Test agent split
3. Commit agent split
4. Implement core/browser.py split (most complex)
5. Test browser split
6. Commit browser split
7. Update documentation

**Files Remaining**: 2 large files (agent_controller.py, browser.py)
**Estimated Effort**: Medium-High (due to complexity and interdependencies)
