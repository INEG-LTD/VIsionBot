from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_agent_events_module():
    module_path = Path(__file__).resolve().parents[2] / "agent" / "events.py"
    spec = spec_from_file_location("agent_events_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_emit_events_field_describes_required_payload_fields() -> None:
    agent_events = _load_agent_events_module()

    field = agent_events.build_emit_events_field(
        [
            agent_events.EventDefinition(
                name="application_file_uploaded",
                description="An application document upload completed.",
                when="after a resume or cover letter file upload succeeds",
                schema={"file_type": str, "file_name": str},
            )
        ]
    )

    description = field["emit_events"]["description"]

    assert "Always include every required field in emit_events[].data exactly as named below." in description
    assert "required_data={file_type:str, file_name:str}" in description
