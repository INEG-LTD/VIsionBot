from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Optional, Sequence


_EVENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class EventDefinition:
    """
    Agent Events policy/config for one named domain event.

    Parameters:
        name:
            Stable event identifier used by tool calls via
            ``emit_events=[{"name": "<name>", "data": {...}}]``.
            Must match ``^[a-z][a-z0-9_]{0,63}$``.
        description:
            Optional human-facing notes shown in planner docs/logs.
        when:
            Optional trigger guidance for the planner (human-readable). This is
            not heuristic-enforced at runtime; runtime enforces only structural
            policy (allowed names/tools, payload shape, required/once/ack rules).
        schema:
            Optional payload contract expressed as ``{field_name: field_type}``.
            Supported field_type values: ``str``, ``int``, ``float``, ``bool``,
            ``dict``, ``list`` (and common aliases like ``string``, ``integer``,
            ``number``, ``boolean``, ``object``, ``array``).
            All keys in schema are treated as required.
        required:
            If True, mission completion is blocked until this event has at least
            one accepted emission in the current mission.
        once_per_mission:
            If True, after the first accepted emission, further emissions for
            this event are rejected for the remainder of the mission.
        terminal:
            If True, an accepted emission ends the current mission immediately.
        require_callback_ack:
            If True, emission is accepted only when callback returns a dict with
            ``{"ack": true}``. Missing callback, callback errors/timeouts, or
            non-ack responses reject the event.
        allowed_tools:
            Optional list of tool names that are allowed to emit this event.
            Empty/None means all tools may emit it.
    """

    name: str
    description: str = ""
    when: str = ""
    schema: Optional[dict[str, Any]] = None
    required: bool = False
    once_per_mission: bool = False
    terminal: bool = False
    require_callback_ack: bool = False
    allowed_tools: Optional[list[str]] = None


@dataclass(frozen=True)
class AgentEvent:
    event_id: str
    action_id: str
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventResult:
    event_id: str
    name: str
    delivered: bool
    response: Any = None
    error: Optional[str] = None


def normalize_event_definitions(
    definitions: Optional[Sequence[EventDefinition]],
) -> tuple[list[EventDefinition], dict[str, EventDefinition]]:
    normalized: list[EventDefinition] = []
    lookup: dict[str, EventDefinition] = {}
    for item in definitions or []:
        if not isinstance(item, EventDefinition):
            continue
        name = str(item.name or "").strip()
        if not _EVENT_NAME_RE.match(name):
            continue
        if name in lookup:
            continue
        when = str(item.when or "").strip()
        desc = str(item.description or "").strip()
        schema = item.schema if isinstance(item.schema, dict) else None
        allowed_tools: list[str] = []
        for tool_name in item.allowed_tools or []:
            text = str(tool_name or "").strip().lower()
            if not text or text in allowed_tools:
                continue
            allowed_tools.append(text)
        definition = EventDefinition(
            name=name,
            when=when,
            schema=schema,
            required=bool(item.required),
            once_per_mission=bool(item.once_per_mission),
            terminal=bool(item.terminal),
            require_callback_ack=bool(item.require_callback_ack),
            allowed_tools=(allowed_tools or None),
            description=desc,
        )
        normalized.append(definition)
        lookup[name] = definition
    return normalized, lookup


def build_emit_events_field(
    definitions: Sequence[EventDefinition],
    *,
    max_items: int = 4,
) -> dict[str, Any]:
    names = [d.name for d in definitions if d.name]
    description_lines = [
        "Emit domain events for this tool action.",
        "Always include every required field in emit_events[].data exactly as named below.",
    ]
    if names:
        description_lines.append("Allowed event names:")
        for d in definitions:
            constraints: list[str] = []
            if d.when:
                constraints.append(f"when={d.when}")
            if d.schema:
                constraints.append(f"required_data={_format_event_schema(d.schema)}")
            if d.required:
                constraints.append("required")
            if d.once_per_mission:
                constraints.append("once_per_mission")
            if d.terminal:
                constraints.append("terminal")
            if d.require_callback_ack:
                constraints.append("require_callback_ack")
            if d.allowed_tools:
                constraints.append(f"allowed_tools={','.join(d.allowed_tools)}")
            line = f"- {d.name}"
            if d.description:
                line += f": {d.description}"
            if constraints:
                line += f" ({'; '.join(constraints)})"
            description_lines.append(line)

    return {
        "emit_events": {
            "type": "array",
            "maxItems": max(1, int(max_items or 1)),
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": names,
                    },
                    "data": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            "description": "\n".join(description_lines),
        }
    }


def _format_event_schema(schema: dict[str, Any]) -> str:
    fields: list[str] = []
    for key, expected in schema.items():
        fields.append(f"{key}:{_describe_event_type(expected)}")
    return "{" + ", ".join(fields) + "}"


def _describe_event_type(expected: Any) -> str:
    if isinstance(expected, str):
        return expected.strip().lower() or "any"
    if isinstance(expected, type):
        return expected.__name__
    if isinstance(expected, tuple) and expected and all(isinstance(item, type) for item in expected):
        return "|".join(item.__name__ for item in expected)
    return "any"


def coerce_emit_events(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    events: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            data = {}
        events.append({"name": name, "data": data})
    return events


def validate_event_payload(definition: EventDefinition, payload: dict[str, Any]) -> tuple[bool, str]:
    schema = definition.schema
    if not isinstance(schema, dict) or not schema:
        return True, ""
    if not isinstance(payload, dict):
        return False, "event payload must be an object"

    for key, expected in schema.items():
        if key not in payload:
            return False, f"missing required payload field '{key}'"
        value = payload.get(key)
        if not _value_matches_type(value, expected):
            return False, f"payload field '{key}' has invalid type"
    return True, ""


def _value_matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        t = expected.strip().lower()
        if t in {"str", "string"}:
            return isinstance(value, str)
        if t in {"int", "integer"}:
            return isinstance(value, int) and not isinstance(value, bool)
        if t in {"float", "number"}:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if t in {"bool", "boolean"}:
            return isinstance(value, bool)
        if t in {"dict", "object"}:
            return isinstance(value, dict)
        if t in {"list", "array"}:
            return isinstance(value, list)
        return True
    if isinstance(expected, type):
        return isinstance(value, expected)
    if isinstance(expected, tuple) and expected and all(isinstance(x, type) for x in expected):
        return isinstance(value, expected)
    return True
