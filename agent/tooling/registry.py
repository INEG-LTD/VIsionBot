from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .policy import EffectPolicyEngine
from .types import (
    BUDGET_FIELDS,
    DialogPolicy,
    NARRATIVE_FIELD,
    NEXT_HINT_FIELD,
    ToolManifest,
    ToolSpec,
)


class ToolRegistry:
    """Runtime registry for declarative tool definitions."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def copy(self) -> "ToolRegistry":
        clone = ToolRegistry()
        clone._tools = dict(self._tools)
        return clone

    def register(self, fn: Any) -> ToolSpec:
        manifest = getattr(fn, "__tool_manifest__", None)
        args_model = getattr(fn, "__tool_args_model__", None)
        if not isinstance(manifest, ToolManifest):
            raise ValueError("Tool is missing ToolManifest metadata. Use @tool(...).")
        if not isinstance(args_model, type) or not issubclass(args_model, BaseModel):
            raise ValueError("Tool is missing a valid args_model BaseModel.")
        signature = inspect.signature(fn)
        params = list(signature.parameters.values())
        if len(params) != 2:
            raise ValueError(
                f"Tool '{getattr(manifest, 'name', '<unknown>')}' must have signature fn(ctx, args)."
            )
        if not manifest.effects:
            raise ValueError(f"Tool '{manifest.name}' must declare at least one effect.")

        name = str(manifest.name or "").strip()
        if not name:
            raise ValueError("Tool manifest name cannot be empty.")
        if name in self._tools:
            raise ValueError(f"Duplicate tool name: {name}")

        description = (manifest.description or "").strip() or (fn.__doc__ or "").strip()
        if not description:
            raise ValueError(f"Tool '{name}' must define description via manifest or docstring.")

        normalized_manifest = replace(manifest, name=name, description=description)
        spec = ToolSpec(manifest=normalized_manifest, args_model=args_model, fn=fn)
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(str(name or "").strip())

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def iter_specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def render_action_text(self, function_name: str, arguments: dict[str, Any]) -> Optional[str]:
        """Render a compact action label for logs/history from registry metadata."""
        spec = self.get(function_name)
        if spec is None:
            return None
        args = dict(arguments or {})
        ignored = {
            "reasoning",
            "narrative",
            "memory_evidence_ids",
            "next_hint_json",
            "budget_spent",
            "budget_remaining",
            "budget_total",
        }
        parts: list[str] = []
        for key, value in args.items():
            if key in ignored or value in (None, "", [], {}):
                continue
            text = str(value)
            if len(text) > 80:
                text = f"{text[:77]}..."
            parts.append(f"{key}={text}")
        if not parts:
            return spec.manifest.name
        return f"{spec.manifest.name}: " + ", ".join(parts[:3])

    def tools_requiring_element(self) -> set[str]:
        names: set[str] = set()
        for name, spec in self._tools.items():
            schema = spec.args_model.model_json_schema()
            props = schema.get("properties") if isinstance(schema, dict) else {}
            if isinstance(props, dict) and "element_id" in props:
                names.add(name)
        return names

    def get_planner_schemas(
        self,
        *,
        dialog_pending: bool,
        budget_enabled: bool,
        policy_engine: Optional[EffectPolicyEngine],
        allowed_names: Optional[List[str]] = None,
        strip_property_descriptions: bool = True,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        allow = {
            str(name).strip()
            for name in (allowed_names or [])
            if str(name).strip()
        }

        for spec in self._tools.values():
            manifest = spec.manifest
            if allow and manifest.name not in allow:
                continue
            if dialog_pending and manifest.dialog_policy == DialogPolicy.BLOCK_WHEN_DIALOG:
                continue
            if policy_engine is not None:
                decision = policy_engine.evaluate(manifest, phase="planner")
                if not decision.allowed and policy_engine.enforce:
                    continue

            params_schema = self._build_parameters_schema(
                spec.args_model,
                budget_enabled=budget_enabled,
                strip_property_descriptions=strip_property_descriptions,
            )
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": manifest.name,
                        "description": manifest.description,
                        "parameters": params_schema,
                    },
                }
            )
        return result

    def _build_parameters_schema(
        self,
        args_model: type[BaseModel],
        *,
        budget_enabled: bool,
        strip_property_descriptions: bool,
    ) -> Dict[str, Any]:
        raw = args_model.model_json_schema()
        schema: Dict[str, Any] = deepcopy(raw if isinstance(raw, dict) else {})
        schema = self._resolve_local_refs(schema)

        # Inline and normalize object schema for OpenAI tools format.
        schema.pop("$defs", None)
        schema.pop("definitions", None)
        schema.pop("title", None)
        schema.setdefault("type", "object")
        properties = schema.setdefault("properties", {})
        if not isinstance(properties, dict):
            properties = {}
            schema["properties"] = properties
        required = schema.setdefault("required", [])
        if not isinstance(required, list):
            required = []
            schema["required"] = required

        properties.update(deepcopy(NARRATIVE_FIELD))
        properties.update(deepcopy(NEXT_HINT_FIELD))
        if "narrative" not in required:
            required.append("narrative")
        if "next_hint_json" not in required:
            required.append("next_hint_json")

        if budget_enabled:
            properties.update(deepcopy(BUDGET_FIELDS))
            for field in BUDGET_FIELDS:
                if field not in required:
                    required.append(field)
        else:
            for field in BUDGET_FIELDS:
                properties.pop(field, None)
            schema["required"] = [item for item in required if item not in BUDGET_FIELDS]

        schema["additionalProperties"] = False

        if strip_property_descriptions:
            props = schema.get("properties")
            if isinstance(props, dict):
                for prop in props.values():
                    if isinstance(prop, dict):
                        prop.pop("description", None)
                        prop.pop("title", None)
        return schema

    def _resolve_local_refs(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        defs = schema.get("$defs")
        if not isinstance(defs, dict):
            defs = schema.get("definitions")
        definitions: Dict[str, Any] = defs if isinstance(defs, dict) else {}

        def _walk(node: Any) -> Any:
            if isinstance(node, dict):
                ref = node.get("$ref")
                if isinstance(ref, str):
                    key = ""
                    if ref.startswith("#/$defs/"):
                        key = ref.split("#/$defs/", 1)[1]
                    elif ref.startswith("#/definitions/"):
                        key = ref.split("#/definitions/", 1)[1]
                    target = definitions.get(key)
                    if isinstance(target, dict):
                        merged = deepcopy(target)
                        for k, v in node.items():
                            if k == "$ref":
                                continue
                            merged[k] = _walk(v)
                        return _walk(merged)
                return {k: _walk(v) for k, v in node.items()}
            if isinstance(node, list):
                return [_walk(item) for item in node]
            return node

        resolved = _walk(deepcopy(schema))
        return resolved if isinstance(resolved, dict) else schema
