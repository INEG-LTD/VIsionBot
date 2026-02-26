"""
Configuration models for Agent.

This module provides structured, type-safe configuration using Pydantic models.
Instead of passing 30+ arguments to Agent, you can create a Config
object with grouped settings.

"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, model_validator
from lib.ai import ReasoningLevel
from browser.provider import BrowserConfig as BrowserProviderConfig


DEFAULT_TOOL_ALLOWLIST: list[str] = [
    "click",
    "type_text",
    "clear_text",
    "select_option",
    "upload_file",
    "set_datetime",
    "press_key",
    "open_url",
    "go_back",
    "go_forward",
    "scroll_down",
    "scroll_up",
    "scroll_container",
    "scroll_to_element",
    "extract_data",
    "ask_user",
    "report_data",
    "write_data",
    "wait_for",
    "send_email",
    "bash",
    "read_file",
    "find_files",
    "read_clipboard",
    "switch_tab",
    "close_tab",
    "open_tab",
    "dismiss_dialog",
    "think",
    "assert_condition",
    "flag",
]

class ToolPreset(str, Enum):
    MINIMAL = "minimal"
    RESEARCH = "research"
    WEB_SAFE = "web_safe"
    FULL = "full"
    LOCKED_DOWN = "locked_down"


TOOL_PRESETS: dict[ToolPreset, list[str]] = {
    # Smallest useful interactive browsing set.
    ToolPreset.MINIMAL: [
        "click",
        "type_text",
        "press_key",
        "open_url",
        "go_back",
        "go_forward",
        "scroll_down",
        "scroll_up",
        "scroll_container",
        "scroll_to_element",
        "wait_for",
        "think",
        "flag",
    ],
    # Practical web research + extraction/reporting, no local shell/fs tools.
    ToolPreset.RESEARCH: [
        "click",
        "type_text",
        "clear_text",
        "select_option",
        "press_key",
        "open_url",
        "go_back",
        "go_forward",
        "scroll_down",
        "scroll_up",
        "scroll_container",
        "scroll_to_element",
        "extract_data",
        "ask_user",
        "report_data",
        "write_data",
        "wait_for",
        "switch_tab",
        "close_tab",
        "open_tab",
        "dismiss_dialog",
        "think",
        "assert_condition",
        "flag",
    ],
    # Browser-only actions; excludes local/host access tools.
    ToolPreset.WEB_SAFE: [
        "click",
        "type_text",
        "clear_text",
        "select_option",
        "upload_file",
        "set_datetime",
        "press_key",
        "open_url",
        "go_back",
        "go_forward",
        "scroll_down",
        "scroll_up",
        "scroll_container",
        "scroll_to_element",
        "extract_data",
        "ask_user",
        "report_data",
        "write_data",
        "wait_for",
        "send_email",
        "switch_tab",
        "close_tab",
        "open_tab",
        "dismiss_dialog",
        "think",
        "assert_condition",
        "flag",
    ],
    # Full legacy default set.
    ToolPreset.FULL: list(DEFAULT_TOOL_ALLOWLIST),
    # Intentional no-op planner sandbox for diagnostics/guardrail checks.
    ToolPreset.LOCKED_DOWN: [
        "ask_user",
        "report_data",
        "write_data",
        "think",
        "flag",
    ],
}


def tool_preset(name: ToolPreset | str) -> list[str]:
    """Resolve a named tool preset to a concrete tool allowlist."""
    preset = (
        name
        if isinstance(name, ToolPreset)
        else ToolPreset(str(name or "").strip().lower())
    )
    # Return a copy to avoid accidental mutation of the shared preset tables.
    return list(TOOL_PRESETS[preset])


def resolve_tool_preset(name: ToolPreset | str) -> tuple[str, list[str]]:
    """Resolve preset id + allowlist for runtime/planner enforcement."""
    preset = (
        name
        if isinstance(name, ToolPreset)
        else ToolPreset(str(name or "").strip().lower())
    )
    return f"preset:{preset.value}", tool_preset(preset)

class ModelConfig(BaseModel):
    """AI model configuration for planning and execution."""

    agent_model: str = Field(
        default="gpt-5-mini",
        description="Model used for high-level agent decisions"
    )
    command_model: str = Field(
        default="gpt-5-mini",
        description="Model used for command generation"
    )
    agent_reasoning_level: ReasoningLevel = Field(
        default=ReasoningLevel.MEDIUM,
        description="Reasoning level for agent decisions"
    )
    command_reasoning_level: ReasoningLevel = Field(
        default=ReasoningLevel.MEDIUM,
        description="Reasoning level for command generation"
    )
    image_detail: str = Field(
        default="high",
        description="Image detail level for vision API: 'low' (faster, cheaper), 'high' (more accurate), or 'auto'"
    )
    
    class Config:
        arbitrary_types_allowed = True


class ExecutionConfig(BaseModel):
    """Runtime execution behavior configuration."""

    max_actions_per_mission: int = Field(
        default=200,
        ge=1,
        description="Maximum number of actions before a mission is forced to end"
    )
    max_actions_per_plan: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of actions to generate in a single action plan. Default is 6. Valid range: 1-20."
    )
    tool_preset: ToolPreset = Field(
        default=ToolPreset.FULL,
        description=(
            "Tool preset used for planner/runtime allowlist enforcement. "
            "Allowed values: minimal, research, web_safe, full, locked_down."
        ),
    )
    budget_constraints_enabled: bool = Field(
        default=True,
        description=(
            "Enable budget-constraining behavior (prompt/schema budget contract, "
            "low-budget single-action planning, and loop-count clamping)."
        ),
    )
    wait_for_load_before_iteration: bool = Field(
        default=False,
        description="If True, wait for the page to reach a load state before each agent iteration."
    )
    wait_for_load_state: str = Field(
        default="networkidle",
        description="Load state to wait for before each agent iteration: 'load', 'domcontentloaded', or 'networkidle'."
    )
    wait_for_load_timeout_ms: int = Field(
        default=30000,
        ge=0,
        description="Max time to wait for page load before each agent iteration (milliseconds)."
    )
    validation_failure_escalation_limit: int = Field(
        default=3,
        ge=0,
        description="Maximum repeated validation failures before escalating to partial/blocked mission result."
    )
    iteration_stall_soft_timeout_s: float = Field(
        default=20.0,
        ge=0.0,
        description=(
            "Soft watchdog timeout in seconds for a single iteration before any action result is produced. "
            "0 disables the soft watchdog."
        ),
    )
    iteration_stall_hard_timeout_s: float = Field(
        default=40.0,
        ge=0.0,
        description=(
            "Hard watchdog timeout in seconds for a single iteration before any action result is produced. "
            "When reached, the current iteration is force-ended and the agent replans. 0 disables hard timeout."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_tool_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        removed_fields = ("tool_profile_id", "tool_profiles", "allowed_tools", "disabled_tools")
        found = [name for name in removed_fields if name in data]
        if not found:
            return data
        raise ValueError(
            "ExecutionConfig no longer supports legacy tool fields: "
            f"{', '.join(found)}. Use tool_preset only."
        )

    class Config:
        arbitrary_types_allowed = True


class ElementConfig(BaseModel):
    """Element detection and overlay configuration."""

    crops_per_gallery: int = Field(
        default=6,
        description="Number of element crops per gallery page in element_index mode. "
                    "Uses 2-column layout with large crops."
    )
    max_index_elements: int = Field(
        default=80,
        description="Maximum number of elements to include in the element_index text. "
                    "Elements are ranked by prominence (area * text_score). "
                    "0 = unlimited."
    )

    class Config:
        arbitrary_types_allowed = True


class DebugConfig(BaseModel):
    """Debugging and logging configuration."""

    debug_mode: bool = Field(
        default=True,
        description="Enable debug mode with verbose logging"
    )
    show_overlay_candidates: bool = Field(
        default=False,
        description="Show detailed overlay candidate information during LLM selection"
    )
    save_screenshots: bool = Field(
        default=False,
        description="Save screenshots sent to the agent for debugging"
    )
    screenshot_dir: str = Field(
        default="agent_screenshots",
        description="Directory to save agent screenshots"
    )
    show_llm_costs: bool = Field(
        default=True,
        description="Show LLM cost information in debug mode"
    )
    telemetry_live_enabled: bool = Field(
        default=True,
        description="Emit per-iteration factual telemetry summaries.",
    )
    telemetry_final_summary_enabled: bool = Field(
        default=True,
        description="Emit mission-final factual telemetry summary.",
    )
    stream_screenshots: bool = Field(
        default=False,
        description="Emit screenshot metadata events and retain screenshots for API retrieval."
    )
    screenshot_stream_persist_to_disk: bool = Field(
        default=True,
        description="Persist streamed screenshots to disk so they remain retrievable after memory eviction."
    )
    screenshot_stream_dir: str = Field(
        default="agent_stream_screenshots",
        description="Directory used by the screenshot stream store."
    )
    screenshot_stream_in_memory_items: int = Field(
        default=40,
        ge=1,
        description="Maximum streamed screenshots retained in memory."
    )
    screenshot_stream_in_memory_mb: int = Field(
        default=120,
        ge=1,
        description="Maximum in-memory screenshot stream size in MB."
    )
    screenshot_stream_max_disk_files: int = Field(
        default=2000,
        ge=0,
        description="Maximum screenshot files retained on disk in stream directory."
    )

    class Config:
        arbitrary_types_allowed = True


class UserInteractionConfig(BaseModel):
    """Policy for interactive user prompts."""

    allow_custom: bool = Field(
        default=True,
        description="Allow user to type a custom answer when choices are presented",
    )
    allow_skip: bool = Field(
        default=True,
        description="Allow user to skip a question (agent is told 'user skipped')",
    )

    class Config:
        arbitrary_types_allowed = True


class SandboxWebConfig(BaseModel):
    """Website policy configuration for browser navigation."""

    allowed_domains: list[str] = Field(
        default_factory=list,
        description=(
            "Allowed domains for web navigation (e.g. 'example.com', '*.example.com'). "
            "When empty, behavior depends on allow_empty_allowlist."
        ),
    )
    allow_empty_allowlist: bool = Field(
        default=True,
        description=(
            "If true, an empty allowed_domains list permits navigation to any website. "
            "If false, an empty allowed_domains list blocks web navigation checks."
        ),
    )

    class Config:
        arbitrary_types_allowed = True


class SandboxFilesystemConfig(BaseModel):
    """Filesystem policy for local tools."""

    allowed_roots: list[str] = Field(
        default_factory=lambda: ["{agent.workspace_root}"],
        description=(
            "Allowed filesystem roots for local read/find operations. "
            "Supports {agent.workspace_root} placeholder."
        ),
    )

    class Config:
        arbitrary_types_allowed = True


class SandboxCommandConfig(BaseModel):
    """Command execution policy for bash tool."""

    allowed_prefixes: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Allowed command prefixes. Example: [['date'], ['rg'], ['git', 'status']]. "
            "Empty list blocks command execution checks (except strict preset defaults)."
        ),
    )
    max_runtime_seconds: int = Field(
        default=30,
        ge=1,
        description="Maximum runtime for a single bash command.",
    )

    class Config:
        arbitrary_types_allowed = True


class SandboxClipboardConfig(BaseModel):
    """Clipboard policy."""

    allow_read: bool = Field(
        default=True,
        description="Allow read_clipboard tool to access system clipboard.",
    )

    class Config:
        arbitrary_types_allowed = True


class SandboxPromptConfig(BaseModel):
    """Prompt policy visibility settings."""

    include_policy_block: bool = Field(
        default=True,
        description="Include sandbox allowlists in planner prompt so the agent knows constraints.",
    )

    class Config:
        arbitrary_types_allowed = True


class SandboxAuditConfig(BaseModel):
    """Run-scoped sandbox audit logging settings."""

    enabled: bool = Field(
        default=True,
        description=(
            "Write sandbox policy decisions to run audit JSONL files when a run audit path is available."
        ),
    )

    class Config:
        arbitrary_types_allowed = True


class SandboxConfig(BaseModel):
    """Sandbox policy configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable sandbox policy checks for local tools and navigation.",
    )
    preset: Literal["trusted", "standard", "strict", "locked"] = Field(
        default="standard",
        description="High-level sandbox preset.",
    )
    mode: Literal["enforce", "observe"] = Field(
        default="enforce",
        description="enforce blocks disallowed actions; observe logs violations but allows execution.",
    )
    web: SandboxWebConfig = Field(
        default_factory=SandboxWebConfig,
        description="Website allowlist policy.",
    )
    fs: SandboxFilesystemConfig = Field(
        default_factory=SandboxFilesystemConfig,
        description="Filesystem allowlist policy.",
    )
    command: SandboxCommandConfig = Field(
        default_factory=SandboxCommandConfig,
        description="Command policy for bash tool.",
    )
    clipboard: SandboxClipboardConfig = Field(
        default_factory=SandboxClipboardConfig,
        description="Clipboard access policy.",
    )
    prompt: SandboxPromptConfig = Field(
        default_factory=SandboxPromptConfig,
        description="Planner prompt policy visibility.",
    )
    audit: SandboxAuditConfig = Field(
        default_factory=SandboxAuditConfig,
        description="Run-scoped sandbox audit logging policy.",
    )

    class Config:
        arbitrary_types_allowed = True


class StorageConfig(BaseModel):
    """Per-agent workspace storage settings."""

    base_dir: str = Field(
        default="bba-data/agents",
        description=(
            "Storage base directory. Runtime normalizes this to an agents root: "
            "if it already ends with 'agents' use it directly, otherwise use <base_dir>/agents."
        ),
    )
    default_persistence_mode: Literal["temp", "persistent"] = Field(
        default="temp",
        description="Default persistence mode for newly created agents.",
    )

    class Config:
        arbitrary_types_allowed = True


class Config(BaseModel):
    """
    Main configuration object for Agent.
    
    This provides a structured, type-safe way to configure the agent instead of
    passing 30+ individual arguments.
    """
    
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="AI model configuration"
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Execution behavior configuration"
    )
    elements: ElementConfig = Field(
        default_factory=ElementConfig,
        description="Element detection configuration"
    )
    logging: DebugConfig = Field(
        default_factory=DebugConfig,
        description="Debug and logging configuration"
    )
    browser: BrowserProviderConfig = Field(
        default_factory=BrowserProviderConfig,
        description="Browser provider configuration"
    )
    user_interaction: UserInteractionConfig = Field(
        default_factory=UserInteractionConfig,
        description="User interaction policy (custom answers, skipping)",
    )
    sandbox: SandboxConfig = Field(
        default_factory=SandboxConfig,
        description="Sandbox policy configuration",
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Per-agent workspace storage configuration",
    )

    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def production(cls) -> Config:
        """
        Create a configuration optimized for production use.
        
        Returns:
            Config with balanced settings for reliability
        """
        return cls(
            execution=ExecutionConfig(
                max_actions_per_mission=1500
            ),
            logging=DebugConfig(debug_mode=False)
        )
    
    @classmethod
    def minimal(cls) -> Config:
        """
        Create a minimal configuration with defaults.
        
        Returns:
            Config with all default settings
        """
        return cls()


_SCHEMA_CONSTRAINT_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "multipleOf",
    "pattern",
    "format",
)


def _humanize_identifier(identifier: str) -> str:
    parts = [part for part in identifier.replace("-", "_").split("_") if part]
    if not parts:
        return identifier
    return " ".join(part.capitalize() for part in parts)


def _resolve_schema_node(
    node: dict[str, Any],
    schema_defs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve local $ref / allOf entries from a model_json_schema tree."""
    if not isinstance(node, dict):
        return {}

    # Typical nested-model form.
    ref = node.get("$ref")
    if isinstance(ref, str):
        prefix = "#/$defs/"
        if ref.startswith(prefix):
            resolved = schema_defs.get(ref[len(prefix):])
            if isinstance(resolved, dict):
                merged = dict(resolved)
                for key, value in node.items():
                    if key != "$ref":
                        merged[key] = value
                return merged

    # Alternate form used when metadata is attached alongside a reference.
    all_of = node.get("allOf")
    if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
        all_of_ref = all_of[0].get("$ref")
        if isinstance(all_of_ref, str):
            prefix = "#/$defs/"
            if all_of_ref.startswith(prefix):
                resolved = schema_defs.get(all_of_ref[len(prefix):])
                if isinstance(resolved, dict):
                    merged = dict(resolved)
                    for key, value in node.items():
                        if key != "allOf":
                            merged[key] = value
                    return merged

    return node


def _schema_type_name(node: dict[str, Any]) -> str:
    type_value = node.get("type")
    if isinstance(type_value, str):
        return type_value
    if isinstance(type_value, list):
        non_null = [value for value in type_value if value != "null"]
        if not non_null:
            return "null"
        return "|".join(dict.fromkeys(non_null))

    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        detected: list[str] = []
        for item in any_of:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "null":
                continue
            if isinstance(item_type, str):
                detected.append(item_type)
            elif "$ref" in item or isinstance(item.get("properties"), dict):
                detected.append("object")
        if detected:
            return "|".join(dict.fromkeys(detected))

    if "$ref" in node or isinstance(node.get("properties"), dict):
        return "object"

    return "unknown"


def _schema_constraints(node: dict[str, Any]) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    for key in _SCHEMA_CONSTRAINT_KEYS:
        if key in node:
            constraints[key] = node[key]

    enum_values = node.get("enum")
    if isinstance(enum_values, list) and enum_values:
        constraints["enum"] = enum_values

    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        nullable = any(
            isinstance(item, dict) and item.get("type") == "null" for item in any_of
        )
        if nullable:
            constraints["nullable"] = True

    return constraints


def _flatten_schema_options(
    schema_node: dict[str, Any],
    schema_defs: dict[str, dict[str, Any]],
    *,
    path_prefix: str = "",
    section: str = "",
    section_name: str = "",
    section_description: str = "",
) -> list[dict[str, Any]]:
    properties = schema_node.get("properties")
    if not isinstance(properties, dict):
        return []

    required_fields = schema_node.get("required")
    required_lookup = set(required_fields) if isinstance(required_fields, list) else set()

    items: list[dict[str, Any]] = []
    for field_name, raw_field_node in properties.items():
        if not isinstance(raw_field_node, dict):
            continue

        field_node = _resolve_schema_node(raw_field_node, schema_defs)
        field_path = f"{path_prefix}.{field_name}" if path_prefix else field_name
        field_display_name = str(
            field_node.get("title") or _humanize_identifier(field_name)
        )
        field_description = str(field_node.get("description") or "")
        is_nested_object = isinstance(field_node.get("properties"), dict)

        if is_nested_object:
            child_section = section or field_name
            child_section_name = section_name or field_display_name
            child_section_description = section_description or field_description
            items.extend(
                _flatten_schema_options(
                    field_node,
                    schema_defs,
                    path_prefix=field_path,
                    section=child_section,
                    section_name=child_section_name,
                    section_description=child_section_description,
                )
            )
            continue

        items.append(
            {
                "path": field_path,
                "section": section or field_name,
                "section_name": section_name or _humanize_identifier(section or field_name),
                "section_description": section_description or field_description,
                "name": field_display_name,
                "description": field_description,
                "type": _schema_type_name(field_node),
                "default": field_node.get("default"),
                "required": field_name in required_lookup,
                "constraints": _schema_constraints(field_node),
            }
        )

    return items


def get_config_option_catalog() -> list[dict[str, Any]]:
    """
    Return flattened metadata for all leaf config options.

    Each entry includes a stable dotted path plus a UI-friendly name and
    description that can be used directly in a settings page.
    """
    schema = Config.model_json_schema()
    schema_defs = schema.get("$defs")
    if not isinstance(schema_defs, dict):
        schema_defs = {}

    options = _flatten_schema_options(schema, schema_defs)
    return sorted(options, key=lambda item: str(item["path"]))


def _section_type_name(annotation: Any) -> str:
    """Best-effort section type name for Config model fields."""
    type_name = getattr(annotation, "__name__", "")
    if type_name:
        return str(type_name)

    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        origin_name = getattr(origin, "__name__", "")
        if origin_name:
            return str(origin_name)

    return str(annotation).replace("typing.", "")


def get_config_section_catalog() -> list[dict[str, str]]:
    """
    Return top-level Config sections with UI metadata.

    This is intended for building section trees such as:
    ModelConfig, ExecutionConfig, ElementConfig, etc.
    """
    items: list[dict[str, str]] = []
    model_fields = getattr(Config, "model_fields", {})
    if not isinstance(model_fields, dict):
        return items

    for section_key, model_field in model_fields.items():
        annotation = getattr(model_field, "annotation", None)
        section_type = _section_type_name(annotation)
        section_description = str(getattr(model_field, "description", "") or "")
        items.append(
            {
                "section": str(section_key),
                "section_type": section_type,
                "section_name": _humanize_identifier(section_key),
                "section_description": section_description,
            }
        )

    return sorted(items, key=lambda item: item["section_type"])
