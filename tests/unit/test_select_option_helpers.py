from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_executor_module():
    return import_module("core.executor.base")


class _FakePage:
    def __init__(self, observed_value: str):
        self._observed_value = observed_value

    def evaluate(self, _script: str, _overlay_index):
        return self._observed_value


def test_dropdown_values_match_accepts_equivalent_display_text() -> None:
    executor_module = _load_executor_module()
    executor = executor_module.Executor.__new__(executor_module.Executor)

    assert executor._dropdown_values_match("United Kingdom", "United Kingdom (+44)") is True
    assert executor._dropdown_values_match("Software Engineer", "Product Manager") is False


def test_read_selected_dropdown_value_returns_none_for_blank_values() -> None:
    executor_module = _load_executor_module()
    executor = executor_module.Executor.__new__(executor_module.Executor)
    executor.browser = SimpleNamespace(page=_FakePage(""))

    assert executor._read_selected_dropdown_value(12) is None
