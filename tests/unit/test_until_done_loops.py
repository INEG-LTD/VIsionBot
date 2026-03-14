from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import threading
import types


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    if module_path.name == "__init__.py":
        spec = spec_from_file_location(
            module_name,
            module_path,
            submodule_search_locations=[str(module_path.parent)],
        )
    else:
        spec = spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_package(package_name: str, relative_path: str):
    package = types.ModuleType(package_name)
    package.__path__ = [str(PROJECT_ROOT / relative_path)]
    sys.modules[package_name] = package
    return package


def _load_loop_modules():
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    _ensure_package("agent", "agent")
    _load_module("agent.tools", "agent/tools/__init__.py")

    cognitive = _load_module("agent.tools.cognitive", "agent/tools/cognitive.py")
    action_planner = _load_module("agent.action_planner", "agent/action_planner.py")
    agent_controller = _load_module("agent.agent_controller", "agent/agent_controller.py")
    return cognitive, action_planner, agent_controller


class _FakeEventLogger:
    def __init__(self) -> None:
        self.loop_events: list[dict] = []
        self.warnings: list[tuple] = []

    def loop_state_changed(self, *args, **kwargs) -> None:
        self.loop_events.append({"args": args, "kwargs": kwargs})

    def system_warning(self, *args, **kwargs) -> None:
        self.warnings.append((args, kwargs))


class _FakeActionExecutor:
    def clear_done_markers(self) -> None:
        return None


def _make_controller_stub(agent_controller_module):
    agent = agent_controller_module.Agent.__new__(agent_controller_module.Agent)
    agent.event_logger = _FakeEventLogger()
    agent._hints_lock = threading.Lock()
    agent._pending_hints = []
    agent._current_iteration = 0
    agent.on_stuck_callback = None
    agent.action_executor = _FakeActionExecutor()
    agent.mission_progress_policy = None
    agent._current_page_metadata = lambda: ("https://example.com/jobs", "Jobs")
    agent._missing_required_agent_events = lambda: set()
    agent._evaluate_finish_attempt = lambda **kwargs: agent_controller_module.FinishDecision(allow=True)
    return agent


def test_think_args_counted_loop_still_requires_loop_count() -> None:
    cognitive, _, _ = _load_loop_modules()
    try:
        cognitive.ThinkArgs(
            reasoning="Need to repeat this action",
            next_action=cognitive.ThinkNextAction.START_LOOP,
            loop_mode="counted",
            loop_description="Fill rows",
        )
    except Exception as exc:
        assert "loop_count is required" in str(exc)
    else:
        raise AssertionError("Expected counted start_loop validation to require loop_count")


def test_think_args_until_done_requires_exit_condition() -> None:
    cognitive, _, _ = _load_loop_modules()
    try:
        cognitive.ThinkArgs(
            reasoning="Keep checking jobs until done",
            next_action=cognitive.ThinkNextAction.START_LOOP,
            loop_mode="until_done",
            loop_description="Inspect distinct jobs",
        )
    except Exception as exc:
        assert "loop_exit_condition is required" in str(exc)
    else:
        raise AssertionError("Expected until_done start_loop validation to require loop_exit_condition")


def test_think_args_until_done_populates_control_fields() -> None:
    cognitive, _, _ = _load_loop_modules()
    args = cognitive.ThinkArgs(
        reasoning="Keep checking jobs until the target is met",
        next_action=cognitive.ThinkNextAction.START_LOOP,
        loop_mode="until_done",
        loop_description="Inspect distinct jobs",
        loop_exit_condition="Stop only when the target is reached or results are exhausted",
    )
    fake_outcome = types.SimpleNamespace(
        output=types.SimpleNamespace(success=True, summary="", error=None, data=None),
        control=None,
    )
    fake_ctx = types.SimpleNamespace(
        runtime_state=types.SimpleNamespace(
            execute_builtin_tool=lambda tool_name, args_model: fake_outcome
        )
    )
    outcome = cognitive.think(fake_ctx, args)

    assert outcome.control is not None
    assert outcome.control.loop_mode == "until_done"
    assert outcome.control.loop_exit_condition == (
        "Stop only when the target is reached or results are exhausted"
    )
    assert outcome.control.completed_rounds == 0


def test_think_args_counted_loop_allows_completed_rounds_before_start() -> None:
    cognitive, _, _ = _load_loop_modules()
    args = cognitive.ThinkArgs(
        reasoning="Resume the remaining review rounds",
        next_action=cognitive.ThinkNextAction.START_LOOP,
        loop_mode="counted",
        loop_count=5,
        completed_rounds=2,
        loop_description="Review rows",
    )

    assert args.completed_rounds == 2


def test_action_planner_renders_until_done_banner_and_prompt_guidance() -> None:
    _, action_planner, _ = _load_loop_modules()
    planner = action_planner.ActionPlanner(
        user_prompt="Collect jobs",
        memory_store=object(),
        in_loop=True,
        loop_round=4,
        loop_mode="until_done",
        loop_description="Inspect distinct Google Jobs",
        loop_exit_condition="Stop only when target is met or results are exhausted",
    )

    reflection = planner._build_reflection_block()
    static_prompt = planner._build_function_calling_static_prompt()

    assert "LOOP 4 (UNTIL DONE): Inspect distinct Google Jobs" in reflection
    assert "Exit condition: Stop only when target is met or results are exhausted" in reflection
    assert 'loop_mode="until_done"' in static_prompt
    assert "Prefer counted loops when you know the number of rounds." in static_prompt


def test_controller_start_loop_until_done_and_advance_keeps_loop_active() -> None:
    _, _, agent_controller = _load_loop_modules()
    agent = _make_controller_stub(agent_controller)
    state = agent_controller.ExecutionState(budget_constraints_enabled=True, budget_remaining=20)
    recent_actions: list[str] = []

    control = agent_controller.ThinkControl(
        next_action=agent_controller.ThinkNextAction.START_LOOP,
        loop_mode="until_done",
        loop_description="Inspect distinct jobs",
        loop_exit_condition="Stop only when the target is met or results are exhausted",
    )

    mission_done, should_replan = agent._apply_think_control(
        control=control,
        state=state,
        append_recent_action=recent_actions.append,
        exit_loop=lambda: None,
    )

    assert mission_done is None
    assert should_replan is True
    assert state.in_loop is True
    assert state.loop_mode == "until_done"
    assert state.loop_count is None
    assert state.loop_round == 1

    state.user_facing_actions_since_progress = 1

    mission_done, should_replan = agent._apply_think_control(
        control=agent_controller.ThinkControl(
            next_action=agent_controller.ThinkNextAction.ADVANCE
        ),
        state=state,
        append_recent_action=recent_actions.append,
        exit_loop=lambda: None,
    )

    assert mission_done is None
    assert should_replan is True
    assert state.in_loop is True
    assert state.loop_round == 2
    assert "until done" in (state.last_action_summary or "").lower()


def test_controller_start_loop_can_resume_after_completed_rounds() -> None:
    _, _, agent_controller = _load_loop_modules()
    agent = _make_controller_stub(agent_controller)
    state = agent_controller.ExecutionState(budget_constraints_enabled=True, budget_remaining=20)

    mission_done, should_replan = agent._apply_think_control(
        control=agent_controller.ThinkControl(
            next_action=agent_controller.ThinkNextAction.START_LOOP,
            loop_mode="counted",
            loop_count=5,
            completed_rounds=2,
            loop_description="Review rows",
        ),
        state=state,
        append_recent_action=lambda _: None,
        exit_loop=lambda: None,
    )

    assert mission_done is None
    assert should_replan is True
    assert state.in_loop is True
    assert state.loop_mode == "counted"
    assert state.loop_count == 5
    assert state.loop_round == 3


def test_controller_counted_loop_still_auto_completes() -> None:
    _, _, agent_controller = _load_loop_modules()
    agent = _make_controller_stub(agent_controller)
    state = agent_controller.ExecutionState(
        in_loop=True,
        loop_mode="counted",
        loop_count=2,
        loop_round=2,
        loop_description="Fill rows",
        user_facing_actions_since_progress=1,
    )
    recent_actions: list[str] = []
    exited = {"value": False}

    def _exit_loop() -> None:
        exited["value"] = True
        state.in_loop = False
        state.loop_count = None
        state.loop_round = 0

    mission_done, should_replan = agent._apply_think_control(
        control=agent_controller.ThinkControl(
            next_action=agent_controller.ThinkNextAction.ADVANCE
        ),
        state=state,
        append_recent_action=recent_actions.append,
        exit_loop=_exit_loop,
    )

    assert mission_done is None
    assert should_replan is True
    assert exited["value"] is True
    assert state.in_loop is False


def test_controller_done_block_keeps_until_done_loop_active_and_tracks_reason() -> None:
    _, _, agent_controller = _load_loop_modules()
    agent = _make_controller_stub(agent_controller)
    agent._evaluate_finish_attempt = lambda **kwargs: agent_controller.FinishDecision(
        allow=False,
        reason="target count not met yet",
        hint="Continue to another unseen job card.",
    )
    state = agent_controller.ExecutionState(
        in_loop=True,
        loop_mode="until_done",
        loop_round=3,
        loop_description="Inspect jobs",
        loop_exit_condition="Stop only when target is reached",
    )

    for _ in range(2):
        mission_done, should_replan = agent._apply_think_control(
            control=agent_controller.ThinkControl(
                next_action=agent_controller.ThinkNextAction.DONE,
                done_reasoning="Done",
            ),
            state=state,
            append_recent_action=lambda _: None,
            exit_loop=lambda: None,
        )
        assert mission_done is None
        assert should_replan is True

    assert state.in_loop is True
    assert state.last_action_summary == "done BLOCKED: target count not met yet"
    assert any("Continue to another unseen job card." in hint for hint in agent._pending_hints)


def test_execution_state_roundtrip_preserves_until_done_fields() -> None:
    _, _, agent_controller = _load_loop_modules()
    state = agent_controller.ExecutionState(
        in_loop=True,
        loop_mode="until_done",
        loop_round=6,
        loop_description="Inspect jobs",
        loop_exit_condition="Stop only when target is reached",
    )

    payload = agent_controller.Agent._execution_state_to_payload(state)
    restored = agent_controller.Agent._execution_state_from_payload(payload)

    assert restored.loop_mode == "until_done"
    assert restored.loop_exit_condition == "Stop only when target is reached"
    assert restored.loop_round == 6
