# Task System Tests

This directory contains comprehensive tests for the sequential task execution system.

## Test Organization

### Unit Tests

**test_task_models.py** - Tests for task data models
- Task type enums
- NormalTask model
- SequentialTask model
- IterationResult model
- TaskList model and helper methods
- BridgePlannerDecision model
- TaskOrchestratorOutput model

**test_task_result_retrieval.py** - Tests for result retrieval system
- TaskResultRetriever keyword matching
- TaskResultRetriever LLM-based matching
- TaskResultAccessor interface
- Result extraction from tasks
- Task dependency handling

**test_bridge_planner.py** - Tests for Bridge Planner
- Retry logic
- Completion strategies (strict, best_effort, threshold)
- Safety limits (max iterations, fail_fast)
- End reason generation
- Iteration history formatting
- Overlay and notebook formatting

### Integration Tests

**test_task_integration.py** - End-to-end integration tests
- Sequential task execution flows
- Task dependency and result flow
- Result retrieval integration
- Completion strategy scenarios
- Error recovery scenarios
- Task list management

### Fixtures

**conftest.py** - Shared pytest fixtures
- Sample tasks (normal and sequential)
- Sample task lists
- Sequential task configurations
- Mock environment state, screenshot, overlay data
- Sample iteration results

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_task_models.py
pytest tests/test_bridge_planner.py
```

### Run Specific Test Class

```bash
pytest tests/test_task_models.py::TestNormalTask
pytest tests/test_bridge_planner.py::TestBridgePlannerRetryLogic
```

### Run Specific Test

```bash
pytest tests/test_task_models.py::TestNormalTask::test_create_normal_task
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=models.task_models --cov=agent.bridge_planner --cov=agent.task_result_retrieval
```

### Run with Coverage Report

```bash
pytest tests/ --cov=models.task_models --cov=agent.bridge_planner --cov-report=html
```

Then open `htmlcov/index.html` in a browser.

## Test Categories

### Model Tests (test_task_models.py)

✅ Task type and status enums
✅ Normal task creation and result storage
✅ Sequential task creation and state tracking
✅ Iteration result tracking
✅ Task list operations (get current, advance, filter)
✅ Task lookup by ID
✅ Completion checking
✅ Bridge Planner decision models

### Retrieval Tests (test_task_result_retrieval.py)

✅ Keyword extraction and matching
✅ Keyword-based task matching
✅ Result extraction from normal tasks
✅ Result extraction from sequential tasks (with None filtering)
✅ Result accessor interface
✅ Task dependency resolution
✅ Empty result handling
✅ Formatting for LLM prompts

### Bridge Planner Tests (test_bridge_planner.py)

✅ Retry logic (under limit, at limit)
✅ Strict completion strategy
✅ Best effort completion strategy
✅ Threshold completion strategy
✅ Indefinite sequences (no target count)
✅ Safety limits (max iterations, fail_fast)
✅ End reason generation
✅ Iteration history formatting (empty, with results, limit)
✅ Overlay summary formatting
✅ Notebook formatting

### Integration Tests (test_task_integration.py)

✅ Successful sequential completion
✅ Sequential with retries
✅ Partial completion scenarios
✅ Task dependency flow
✅ Multi-task result flow
✅ Result retrieval after execution
✅ Completion strategy scenarios
✅ Error recovery scenarios
✅ Task list progression

## Test Coverage Goals

- **Models**: 100% coverage of all task models
- **Bridge Planner**: 95%+ coverage of decision logic
- **Result Retrieval**: 90%+ coverage of retrieval methods
- **Integration**: Cover all major execution paths

## Adding New Tests

When adding new functionality:

1. **Add unit tests** for new functions/methods in the appropriate test file
2. **Add integration tests** if the feature spans multiple components
3. **Update fixtures** in conftest.py if new test data is needed
4. **Document** new test categories in this README

### Test Naming Convention

- Test files: `test_<module_name>.py`
- Test classes: `Test<FeatureName>`
- Test methods: `test_<specific_behavior>`

### Example

```python
class TestBridgePlannerRetryLogic:
    """Test Bridge Planner retry logic"""

    def test_should_retry_iteration_under_limit(self):
        """Test that retry is allowed when under limit"""
        # Test implementation
        pass
```

## Continuous Integration

Tests should be run in CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ --cov --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Debugging Tests

### Run with print output

```bash
pytest tests/ -s
```

### Run with debugging on failure

```bash
pytest tests/ --pdb
```

### Run only failed tests

```bash
pytest tests/ --lf
```

### Run with markers

```bash
# Run only slow tests
pytest tests/ -m slow

# Run everything except slow tests
pytest tests/ -m "not slow"
```

## Known Issues / Limitations

- LLM-based matching tests are mocked (no actual API calls)
- Some integration tests may need actual browser instance for full E2E
- Task execution wiring tests pending (requires ReactiveGoalDeterminer mocking)

## Future Test Additions

- [ ] Task orchestrator decomposition tests
- [ ] Full E2E tests with mocked browser
- [ ] Performance benchmarks
- [ ] Stress tests (100+ iterations)
- [ ] Concurrent execution tests
- [ ] Result persistence tests
