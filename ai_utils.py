"""Shared utilities for invoking chat/vision models through LiteLLM.

The helpers in this module provide a consistent way to:
    * keep default model / reasoning settings in sync
    * map model identifiers to providers and API keys
    * call LiteLLM with optional vision inputs and schema parsing
    * obtain usage and cost information in a common format
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Type, Union, List
from collections.abc import Sequence as SequenceABC
import re

from litellm import completion, completion_cost
try:
    from litellm.exceptions import UnsupportedParamsError
except ImportError:
    # Fallback if the exception isn't available
    UnsupportedParamsError = None

import litellm
litellm.suppress_debug_info = True

from pydantic import BaseModel, ValidationError


class ReasoningLevel(str, Enum):
    """Supported reasoning effort knobs for providers that expose them."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def coerce(cls, value: Union["ReasoningLevel", str]) -> "ReasoningLevel":
        """
        Return a `ReasoningLevel` instance for the provided value.

        Examples
        --------
        >>> ReasoningLevel.coerce("LOW")
        <ReasoningLevel.LOW: 'low'>
        >>> ReasoningLevel.coerce(ReasoningLevel.HIGH)
        <ReasoningLevel.HIGH: 'high'>
        >>> ReasoningLevel.coerce("Expert")
        Traceback (most recent call last):
            ...
        ValueError: Invalid reasoning level 'expert'. Allowed values: none, low, medium, high.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(level.value for level in cls)
            raise ValueError(
                f"Invalid reasoning level '{value}'. Allowed values: {allowed}."
            ) from exc


# Global defaults – kept in module state so callers can change them centrally.
_DEFAULT_MODEL = "gpt-5-mini"
_DEFAULT_REASONING_LEVEL: str = ReasoningLevel.MEDIUM.value
_DEFAULT_AGENT_MODEL = "gpt-5-mini"
_DEFAULT_AGENT_REASONING_LEVEL: str = ReasoningLevel.MEDIUM.value


# Optional per-model or per-provider API keys. Populate this if you do not want
# to rely solely on environment variables.
MODEL_API_KEYS: dict[str, str] = {}


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration describing provider-specific behaviour."""

    env_vars: Sequence[str]
    supports_reasoning_flag: bool = False
    requires_string_content: bool = False
    supports_image_understanding: bool = True


_PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(env_vars=("OPENAI_API_KEY",)),
    "google": ProviderConfig(
        env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        supports_reasoning_flag=True,
    ),
    "gemini": ProviderConfig(
        env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        supports_reasoning_flag=True,
    ),
    "groq": ProviderConfig(
        env_vars=("GROQ_API_KEY",),
        requires_string_content=True,
    ),
    "anthropic": ProviderConfig(
        env_vars=("ANTHROPIC_API_KEY",),
        supports_reasoning_flag=True,
    ),
    "deepseek": ProviderConfig(
        env_vars=("DEEPSEEK_API_KEY",),
        supports_reasoning_flag=True,
    ),
    "cerebras": ProviderConfig(
        env_vars=("CEREBRAS_API_KEY",),
        supports_reasoning_flag=False,
        supports_image_understanding=False,
    ),
}

__all__ = [
    "ReasoningLevel",
    "MODEL_API_KEYS",
    "set_default_model",
    "get_default_model",
    "set_default_agent_model",
    "get_default_agent_model",
    "set_default_reasoning_level",
    "get_default_reasoning_level",
    "set_default_agent_reasoning_level",
    "get_default_agent_reasoning_level",
    "generate_text_with_cost",
    "generate_text_gpt_with_cost",
    "generate_text",
    "answer_question_with_vision",
    "generate_model_with_cost",
    "generate_model_gpt_with_cost",
    "generate_model_gpt",
    "generate_model",
]


# ---------------------------------------------------------------------------
# Public configuration helpers
# ---------------------------------------------------------------------------


def set_default_model(model_name: str) -> None:
    """Update the fallback model used by helpers when none is provided."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = model_name


def get_default_model() -> str:
    """Return the globally configured default model name."""
    return _DEFAULT_MODEL


def set_default_agent_model(model_name: str) -> None:
    """Update the fallback agent model used by helpers when none is provided."""
    global _DEFAULT_AGENT_MODEL
    _DEFAULT_AGENT_MODEL = model_name


def get_default_agent_model() -> str:
    """Return the globally configured default agent model name."""
    return _DEFAULT_AGENT_MODEL


def set_default_reasoning_level(reasoning_level: Union[ReasoningLevel, str]) -> None:
    """Set the default reasoning level used when callers omit the parameter."""
    global _DEFAULT_REASONING_LEVEL
    _DEFAULT_REASONING_LEVEL = ReasoningLevel.coerce(reasoning_level).value


def get_default_reasoning_level() -> str:
    """Return the globally configured default reasoning level."""
    return _DEFAULT_REASONING_LEVEL


def set_default_agent_reasoning_level(
    reasoning_level: Union[ReasoningLevel, str]
) -> None:
    """Set the default agent reasoning level used when callers omit the parameter."""
    global _DEFAULT_AGENT_REASONING_LEVEL
    _DEFAULT_AGENT_REASONING_LEVEL = ReasoningLevel.coerce(reasoning_level).value


def get_default_agent_reasoning_level() -> str:
    """Return the globally configured default agent reasoning level."""
    return _DEFAULT_AGENT_REASONING_LEVEL


# ---------------------------------------------------------------------------
# Internal helper utilities
# ---------------------------------------------------------------------------


def _infer_provider(model: str) -> str:
    """Best-effort provider inference based on model id."""
    if "/" in model:
        candidate = model.split("/", 1)[0].lower()
        if candidate in _PROVIDERS:
            return candidate
    lowered = model.lower()
    if "gemini" in lowered or "google" in lowered:
        return "google"
    if "groq" in lowered:
        return "groq"
    if "deepseek" in lowered:
        return "deepseek"
    if "anthropic" in lowered or "claude" in lowered:
        return "anthropic"
    return "openai"


# Models that don't support reasoning_effort parameter
_MODELS_WITHOUT_REASONING: set[str] = {
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-8b-001",
    # Add more models as needed
}


def _model_supports_reasoning(model: str, provider: str, config: ProviderConfig) -> bool:
    """
    Check if a specific model supports the reasoning_effort parameter.
    
    Args:
        model: The model identifier
        provider: The provider name
        config: The provider configuration
    
    Returns:
        True if the model supports reasoning, False otherwise
    """
    # First check if provider supports reasoning at all
    if not config.supports_reasoning_flag:
        return False
    
    # Check model-specific exceptions
    model_lower = model.lower()
    if model_lower in _MODELS_WITHOUT_REASONING:
        return False
    
    # Check for patterns in model names that indicate no reasoning support
    # Some "lite" or "flash" models may not support reasoning
    if provider == "google" or provider == "gemini":
        # Most Gemini models support reasoning, but some lite/flash variants don't
        if "-lite" in model_lower or "-flash" in model_lower:
            # Check against known exceptions
            if model_lower not in _MODELS_WITHOUT_REASONING:
                # For unknown lite/flash models, we'll try and catch the error
                # This allows us to be permissive but handle failures gracefully
                pass
    
    return True


def _resolve_api_key(model: str, provider: str) -> str:
    """Locate an API key for the provider/model combination."""
    override = MODEL_API_KEYS.get(model) or MODEL_API_KEYS.get(provider)
    if override:
        return override

    config = _PROVIDERS.get(provider)
    env_names = config.env_vars if config else ()
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value

    env_hint = ", ".join(env_names) or "<provider specific env var>"
    raise RuntimeError(
        f"No API key configured for model '{model}' (provider '{provider}'). "
        f"Set one in MODEL_API_KEYS or via environment variable(s): {env_hint}."
    )


def _prepare_image_part(image: Union[bytes, bytearray, str], image_detail: str) -> dict[str, Any]:
    """Convert bytes/base64 content into the structure expected by LiteLLM."""
    if isinstance(image, (bytes, bytearray)):
        b64 = base64.b64encode(image).decode("ascii")
        url = f"data:image/jpeg;base64,{b64}"
    else:
        string_value = str(image)
        url = string_value if string_value.startswith("data:image/") else f"data:image/jpeg;base64,{string_value}"
    return {"type": "image_url", "image_url": {"url": url, "detail": image_detail}}


def _build_user_content(
    prompt: str,
    image: Optional[Union[bytes, bytearray, str]],
    multi_image: Optional[Sequence[bytes]],
    image_detail: str,
    *,
    requires_string_content: bool,
    supports_image_understanding: bool = True,
) -> Union[str, list[dict[str, Any]]]:
    """Produce the user message content, honouring provider content rules."""
    if requires_string_content:
        segments: list[str] = []
        if prompt:
            segments.append(prompt)
        attachments = (len(multi_image) if multi_image else 0) + (1 if image is not None else 0)
        if attachments:
            segments.append(f"[{attachments} image(s) attached – omitted for this provider]")
        return "\n\n".join(segments)

    content_parts: list[dict[str, Any]] = []
    if prompt:
        content_parts.append({"type": "text", "text": prompt})
    # Only add images if the provider supports image understanding
    if supports_image_understanding:
        if multi_image:
            for extra_image in multi_image:
                content_parts.append(_prepare_image_part(extra_image, image_detail))
        if image is not None:
            content_parts.append(_prepare_image_part(image, image_detail))
    return content_parts or [{"type": "text", "text": prompt}]


def _build_messages_for_provider(
    prompt: str,
    system_prompt: str,
    image: Optional[Union[bytes, bytearray, str]],
    multi_image: Optional[Sequence[bytes]],
    image_detail: str,
    config: ProviderConfig,
) -> list[dict[str, Any]]:
    """Compose the LiteLLM messages payload for the given provider."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt if not config.requires_string_content else system_prompt,
            }
        )

    user_content = _build_user_content(
        prompt=prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        requires_string_content=config.requires_string_content,
        supports_image_understanding=config.supports_image_understanding,
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def _extract_text_from_response(response: dict[str, Any]) -> str:
    """Best-effort text extraction from a LiteLLM completion response."""
    choices = response.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    parsed = message.get("parsed")
    if parsed is not None:
        return parsed if isinstance(parsed, str) else str(parsed)

    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(segment.get("text", "") for segment in content if isinstance(segment, dict) and segment.get("text"))
    if isinstance(content, str):
        return content
    return ""


def _extract_usage(response: dict[str, Any], model: str) -> Tuple[float, dict[str, Any]]:
    """Extract token usage and cost data from a LiteLLM response."""
    usage = response.get("usage") or {}
    input_tokens = usage.get("prompt_tokens") or usage.get("prompt_tokens_total") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("completion_tokens_total") or usage.get("output_tokens") or 0
    total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens) or None

    try:
        cost_usd = completion_cost(response) or 0.0
    except Exception:
        cost_usd = 0.0

    return cost_usd, {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _build_response_format(model_object_type: Optional[Type[BaseModel]], model: str = "") -> dict[str, Any]:
    """Construct a LiteLLM-compatible response schema when structured output is requested."""
    if model_object_type is None:
        return {"type": "json_object"}
    
    # Gemini models can use json_schema mode with some limitations
    if "gemini" in model.lower() or "google" in model.lower():
        try:
            schema = model_object_type.model_json_schema()
        except AttributeError:
            schema = model_object_type.schema()
        _enforce_no_additional_properties(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model_object_type.__name__,
                "schema": schema,
                "strict": False,  # Less strict for Gemini compatibility
            },
        }
    
    try:
        schema = model_object_type.model_json_schema()
    except AttributeError:
        schema = model_object_type.schema()
    _enforce_no_additional_properties(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_object_type.__name__,
            "schema": schema,
            "strict": True,
        },
    }


def _enforce_no_additional_properties(schema: Any) -> None:
    """Recursively set additionalProperties=false on object schemas."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        props = schema.get("properties", {})
        if props:
            schema["required"] = list(props.keys())
        for prop in schema.get("properties", {}).values():
            _enforce_no_additional_properties(prop)
    if "items" in schema:
        _enforce_no_additional_properties(schema["items"])
    if "anyOf" in schema:
        for entry in schema["anyOf"]:
            _enforce_no_additional_properties(entry)
    if "allOf" in schema:
        for entry in schema["allOf"]:
            _enforce_no_additional_properties(entry)
    if "oneOf" in schema:
        for entry in schema["oneOf"]:
            _enforce_no_additional_properties(entry)


def _extract_json_object(text: str) -> Optional[Any]:
    """Best-effort extraction of a JSON object embedded in arbitrary text."""
    if not isinstance(text, str):
        return None

    candidates: List[str] = []
    stripped = text.strip()
    if stripped.startswith("```"):
        for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
            candidates.append(block)
    candidates.append(stripped)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate_stripped = candidate.strip()
        try:
            return json.loads(candidate_stripped)
        except Exception:
            pass

        for idx, ch in enumerate(candidate_stripped):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(candidate_stripped[idx:])
                    return obj
                except Exception:
                    continue
    return None


def _manual_parse_structured_output(text: str, model_object_type: Type[BaseModel]) -> Optional[BaseModel]:
    """Attempt lightweight manual parsing when the model fails to return valid JSON."""
    try:
        if model_object_type.__name__ == "NextAction":
            return _manual_parse_next_action(text, model_object_type)
        if model_object_type.__name__ == "CompletionEvaluation":
            return _manual_parse_completion_evaluation(text, model_object_type)
        if model_object_type.__name__ == "ExtractionResult":
            return _manual_parse_extraction_result(text, model_object_type)
        
        # Try to parse simple yes/no boolean responses like "is_complete (false)" or "needs_sub_agents false"
        cleaned = text.strip()
        
        # First, try to match "field_name: value" patterns
        simple_bool_match = re.match(r'^(\w+)\s*[\(:]?\s*(true|false|yes|no)\s*[\)]?$', cleaned, re.IGNORECASE)
        if simple_bool_match:
            field_name = simple_bool_match.group(1)
            value_str = simple_bool_match.group(2).lower()
            bool_value = value_str in ('true', 'yes')
            try:
                if hasattr(model_object_type, "model_validate"):
                    return model_object_type.model_validate({field_name: bool_value})
                return model_object_type(**{field_name: bool_value})
            except ValidationError as exc:
                print(f"⚠️ Simple boolean parse failed validation for {model_object_type.__name__}: {exc}")
        
        # Handle plain boolean text like "false" or "true" by inferring field name from schema
        if cleaned.lower() in ('true', 'false', 'yes', 'no'):
            bool_value = cleaned.lower() in ('true', 'yes')
            try:
                # Try to get the schema to find boolean field names
                schema = model_object_type.model_json_schema() if hasattr(model_object_type, "model_json_schema") else model_object_type.schema()
                properties = schema.get('properties', {})
                # Find boolean fields
                bool_fields = [name for name, prop in properties.items() if prop.get('type') == 'boolean']
                if len(bool_fields) == 1:
                    # If there's only one boolean field, use it
                    field_name = bool_fields[0]
                    if hasattr(model_object_type, "model_validate"):
                        return model_object_type.model_validate({field_name: bool_value})
                    return model_object_type(**{field_name: bool_value})
            except Exception as exc:
                print(f"⚠️ Plain boolean inference failed for {model_object_type.__name__}: {exc}")
        
        # Generic fallback: try to extract JSON from markdown-wrapped responses
        json_obj = _extract_json_object(text)
        if isinstance(json_obj, dict):
            try:
                if hasattr(model_object_type, "model_validate"):
                    return model_object_type.model_validate(json_obj)
                return model_object_type(**json_obj)
            except ValidationError as exc:
                print(f"⚠️ Generic JSON extraction failed validation for {model_object_type.__name__}: {exc}")
                return None
    except Exception as exc:
        print(f"⚠️ Manual parse helper error for {model_object_type.__name__}: {exc}")
    return None


def _manual_parse_next_action(text: str, model_object_type: Type[BaseModel]) -> Optional[BaseModel]:
    """
    Extract a NextAction response from loosely structured prose.
    Accepts free-form text like:
        "Action: click: Google Search button\nReasoning: ...\nConfidence: 0.8\nExpected outcome: ...\nNeeds exploration: no"
    Falls back to heuristics if explicit keys are missing.
    """
    cleaned = text.strip()
    if not cleaned:
        return None

    json_obj = _extract_json_object(cleaned)
    if isinstance(json_obj, dict):
        if isinstance(json_obj.get("evidence"), dict):
            try:
                json_obj["evidence"] = json.dumps(json_obj["evidence"])
            except Exception:
                json_obj["evidence"] = str(json_obj["evidence"])
        try:
            if hasattr(model_object_type, "model_validate"):
                return model_object_type.model_validate(json_obj)  # type: ignore[attr-defined]
            return model_object_type(**json_obj)
        except ValidationError as exc:
            print(f"⚠️ Manual NextAction JSON parse failed validation: {exc}")
        except Exception as exc:
            print(f"⚠️ Manual NextAction JSON parse error: {exc}")

    # Try explicit key-value extraction first.
    pairs = {}
    for key in ("action", "reasoning", "confidence", "expected outcome", "expected_outcome", "needs exploration", "needs_exploration"):
        match = re.search(rf"{key}\s*[:\-]\s*(.+)", cleaned, flags=re.IGNORECASE)
        if match:
            pairs[key.lower().replace(" ", "_")] = match.group(1).strip(" \n\r\t\"'")

    action = pairs.get("action")
    if not action:
        # Fallback: grab the first command-like phrase (click:, type:, scroll:, press:, wait:, open:, back:, forward:, stop:)
        cmd_match = re.search(
            r"\b(click|type|scroll|press|wait|open|back|forward|stop|handle_select|upload|handle_datetime)\s*:?\s*[^\n\.]+",
            cleaned,
            flags=re.IGNORECASE,
        )
        if cmd_match:
            action = cmd_match.group(0).strip(" .")
    if not action:
        return None

    reasoning = pairs.get("reasoning") or cleaned

    confidence_str = pairs.get("confidence")
    confidence = 0.5
    if confidence_str:
        try:
            confidence = float(confidence_str)
        except ValueError:
            pass
    confidence = max(0.0, min(confidence, 1.0))

    expected_outcome = pairs.get("expected_outcome") or pairs.get("expected outcome") or ""
    if not expected_outcome:
        # Try to infer expectation from prose after "so that"/"to"
        outcome_match = re.search(r"(?:so that|to)\s+([^\.]+)", cleaned, flags=re.IGNORECASE)
        if outcome_match:
            expected_outcome = outcome_match.group(1).strip()
    if not expected_outcome:
        expected_outcome = reasoning

    needs_exploration_raw = pairs.get("needs_exploration") or pairs.get("needs exploration") or ""
    needs_exploration = needs_exploration_raw.lower() in {"yes", "true", "1"} if needs_exploration_raw else False
    if not needs_exploration and "explor" in reasoning.lower():
        needs_exploration = True

    try:
        return model_object_type(
            action=action.strip(),
            reasoning=reasoning.strip(),
            confidence=confidence,
            expected_outcome=expected_outcome.strip(),
            needs_exploration=needs_exploration,
        )
    except ValidationError as exc:
        print(f"⚠️ Manual NextAction parse failed validation: {exc}")
        return None


def _manual_parse_completion_evaluation(text: str, model_object_type: Type[BaseModel]) -> Optional[BaseModel]:
    """
    Heuristic parser for CompletionEvaluation when the model emits prose.
    """
    cleaned = text.strip()
    if not cleaned:
        return None

    json_obj = _extract_json_object(cleaned)
    if isinstance(json_obj, dict):
        def _coerce_fields(data: dict[str, Any]) -> None:
            if isinstance(data.get("evidence"), dict):
                try:
                    data["evidence"] = json.dumps(data["evidence"])
                except Exception:
                    data["evidence"] = str(data["evidence"])
            if isinstance(data.get("remaining_steps"), str):
                stripped = data["remaining_steps"].strip()
                if stripped:
                    data["remaining_steps"] = [stripped]
                else:
                    data["remaining_steps"] = []

        _coerce_fields(json_obj)
        try:
            if hasattr(model_object_type, "model_validate"):
                return model_object_type.model_validate(json_obj)  # type: ignore[attr-defined]
            return model_object_type(**json_obj)
        except ValidationError as exc:
            _coerce_fields(json_obj)
            try:
                if hasattr(model_object_type, "model_validate"):
                    return model_object_type.model_validate(json_obj)  # type: ignore[attr-defined]
                return model_object_type(**json_obj)
            except Exception:
                print(f"⚠️ Manual CompletionEvaluation JSON parse failed validation: {exc}")
        except Exception as exc:
            print(f"⚠️ Manual CompletionEvaluation JSON parse error: {exc}")

    lower = cleaned.lower()

    # Determine completion status heuristically
    negative_patterns = [
        r"\bnot\s+(?:yet\s+)?complete\b",
        r"\bnot\s+(?:yet\s+)?completed\b",
        r"\bnot\s+(?:yet\s+)?fulfilled\b",
        r"\bnot\s+(?:yet\s+)?done\b",
        r"\bnot\s+(?:yet\s+)?achieved\b",
        r"\bnot\s+(?:yet\s+)?finished\b",
        r"\bincomplete\b",
        r"\bfailed\b",
        r"\bstill\s+in\s+progress\b",
        r"\bhas\s+not\s+been\s+fulfilled\b",
        r"\bhas\s+not\s+been\s+achieved\b",
    ]
    positive_pattern = r"(?<!not\s)(?:task\s+)?(?:complete|completed|fulfilled|done|achieved|success(?:fully)?)"

    is_complete = False
    if any(re.search(pattern, lower) for pattern in negative_patterns):
        is_complete = False
    elif re.search(positive_pattern, lower):
        is_complete = True

    # Confidence
    confidence = 0.5
    conf_match = re.search(r"confidence\s*[:\-]\s*([0-9]*\.?[0-9]+)", cleaned, flags=re.IGNORECASE)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            pass
    else:
        conf_pct = re.search(r"confidence\s*[:\-]\s*([0-9]{1,3})\s*%", cleaned, flags=re.IGNORECASE)
        if conf_pct:
            try:
                confidence = float(conf_pct.group(1)) / 100.0
            except ValueError:
                pass
    confidence = max(0.0, min(confidence, 1.0))

    reasoning = cleaned
    reasoning_match = re.search(r"(reasoning|analysis|explanation)\s*[:\-]\s*(.+)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(2).strip()

    evidence = None
    evidence_match = re.search(r"(evidence|support|proof)\s*[:\-]\s*(.+)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if evidence_match:
        evidence = evidence_match.group(2).strip()

    remaining_steps: List[str] = []
    rem_section = re.search(r"(remaining\s+steps|next\s+steps|todo|actions)\s*[:\-]\s*(.+)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if rem_section:
        block = rem_section.group(2)
        lines = [line.strip(" -*\t\r") for line in block.splitlines()]
        for line in lines:
            if not line:
                continue
            if re.match(r"(reason|confidence|evidence)\s*[:\-]", line, flags=re.IGNORECASE):
                break
            remaining_steps.append(line)

    try:
        return model_object_type(
            is_complete=is_complete,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            remaining_steps=remaining_steps,
        )
    except ValidationError as exc:
        print(f"⚠️ Manual CompletionEvaluation parse failed validation: {exc}")
        return None


def _manual_parse_extraction_result(text: str, model_object_type: Type[BaseModel]) -> Optional[BaseModel]:
    """
    Heuristic parser for ExtractionResult when the model emits prose instead of structured JSON.
    
    Attempts to extract:
    - extracted_data: JSON object or key-value pairs
    - confidence: float between 0.0 and 1.0
    - reasoning: explanation text
    """
    import json
    
    cleaned = text.strip()
    if not cleaned:
        return None
    
    # First, try to extract JSON object from the text
    json_obj = _extract_json_object(cleaned)
    if isinstance(json_obj, dict):
        # Try to match ExtractionResult structure
        extracted_data = json_obj.get("extracted_data") or json_obj.get("data") or json_obj
        confidence = json_obj.get("confidence", 0.7)
        reasoning = json_obj.get("reasoning") or json_obj.get("explanation") or "Extracted from JSON response"
        
        # Ensure extracted_data is a JSON string
        if isinstance(extracted_data, dict):
            extracted_data = json.dumps(extracted_data)
        elif not isinstance(extracted_data, str):
            extracted_data = json.dumps({"content": str(extracted_data)})
        
        try:
            # Validate confidence
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
            
            if hasattr(model_object_type, "model_validate"):
                return model_object_type.model_validate({
                    "extracted_data": extracted_data,
                    "confidence": confidence,
                    "reasoning": reasoning
                })
            return model_object_type(
                extracted_data=extracted_data,
                confidence=confidence,
                reasoning=reasoning
            )
        except (ValidationError, ValueError) as exc:
            print(f"⚠️ Manual ExtractionResult JSON parse failed validation: {exc}")
    
    # Fallback: extract key-value pairs from prose
    pairs = {}
    for key in ("extracted_data", "extracted data", "data", "confidence", "reasoning", "explanation"):
        match = re.search(rf"{key}\s*[:\-]\s*(.+)", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match:
            pairs[key.lower().replace(" ", "_")] = match.group(1).strip(" \n\r\t\"'")
    
    # Try to extract JSON from the text content
    extracted_data_raw = pairs.get("extracted_data") or pairs.get("data") or cleaned
    extracted_data = None
    
    # Try to find JSON in the text
    json_obj = _extract_json_object(extracted_data_raw)
    if json_obj:
        extracted_data = json.dumps(json_obj)
    else:
        # If no JSON found, wrap the content as a description
        # Remove common prefixes like "extracted_data:", "data:", etc.
        content = extracted_data_raw
        for prefix in ["extracted_data:", "data:", "extraction:", "result:"]:
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()
        # Try to create a simple key-value structure from the description
        extracted_data = json.dumps({"description": content[:500]})
    
    # Extract confidence
    confidence = 0.7  # Default
    confidence_str = pairs.get("confidence")
    if confidence_str:
        try:
            confidence = float(re.search(r"[\d.]+", confidence_str).group() if re.search(r"[\d.]+", confidence_str) else confidence_str)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, AttributeError):
            pass
    
    # Extract reasoning
    reasoning = pairs.get("reasoning") or pairs.get("explanation") or cleaned[:200]
    
    try:
        if hasattr(model_object_type, "model_validate"):
            return model_object_type.model_validate({
                "extracted_data": extracted_data,
                "confidence": confidence,
                "reasoning": reasoning
            })
        return model_object_type(
            extracted_data=extracted_data,
            confidence=confidence,
            reasoning=reasoning
        )
    except ValidationError as exc:
        print(f"⚠️ Manual ExtractionResult parse failed validation: {exc}")
        return None


def _perform_completion(
    *,
    prompt: str,
    system_prompt: str,
    image: Optional[Union[bytes, bytearray, str]],
    multi_image: Optional[Sequence[bytes]],
    image_detail: str,
    model: str,
    reasoning_level: Optional[str],
    response_format: Optional[dict[str, Any]] = None,
) -> Tuple[str, float, dict[str, Any], dict[str, Any]]:
    """Invoke LiteLLM and return the text, usage, cost, and raw response."""
    provider = _infer_provider(model)
    config = _PROVIDERS.get(provider, ProviderConfig(env_vars=()))
    api_key = _resolve_api_key(model, provider)
    messages = _build_messages_for_provider(
        prompt=prompt,
        system_prompt=system_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        config=config,
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "api_key": api_key,
        "tool_choice": "none",
        "tools": [],
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    
    # Check if model supports reasoning before adding the parameter
    use_reasoning = reasoning_level and _model_supports_reasoning(model, provider, config)
    if use_reasoning:
        kwargs["reasoning_effort"] = reasoning_level

    # Try completion with reasoning, fallback without if it fails
    try:
        response = completion(**kwargs)
    except Exception as e:
        # If we get an UnsupportedParamsError for reasoning_effort, retry without it
        error_str = str(e).lower()
        is_reasoning_error = (
            (UnsupportedParamsError is not None and isinstance(e, UnsupportedParamsError)) or
            (("reasoning_effort" in error_str or "reasoning" in error_str) and
             ("unsupported" in error_str or "not support" in error_str or "does not support" in error_str))
        )
        
        if is_reasoning_error and "reasoning_effort" in kwargs:
            # Remove reasoning_effort and retry
            kwargs.pop("reasoning_effort", None)
            print(f"⚠️ Model {model} doesn't support reasoning_effort, retrying without it...")
            # Retry
            response = completion(**kwargs)
            # Cache this model as not supporting reasoning for future calls
            _MODELS_WITHOUT_REASONING.add(model.lower())
        else:
            raise
    text = _extract_text_from_response(response)
    cost_usd, usage = _extract_usage(response, model)

    try:
        from utils.event_logger import get_event_logger
        get_event_logger().llm_cost(
            cost_usd=cost_usd,
            input_tokens=usage['input_tokens'],
            output_tokens=usage['output_tokens'],
            total_tokens=usage['total_tokens'],
            model=model
        )
    except Exception:
        pass

    return text, cost_usd, usage, response


def _normalize_reasoning_level(
    reasoning_level: Union[ReasoningLevel, str, None]
) -> tuple[ReasoningLevel, Optional[str]]:
    """Normalize caller input to `(enum, provider_value)`."""
    if reasoning_level is None:
        reasoning_enum = ReasoningLevel.coerce(get_default_reasoning_level())
    else:
        reasoning_enum = ReasoningLevel.coerce(reasoning_level)

    if reasoning_enum is ReasoningLevel.NONE:
        return reasoning_enum, None
    return reasoning_enum, reasoning_enum.value


# ---------------------------------------------------------------------------
# Public text-generation helpers
# ---------------------------------------------------------------------------


def generate_text_with_cost(
    prompt: str,
    system_prompt: str = "",
    image: bytes | bytearray | str | None = None,
    multi_image: Optional[Sequence[bytes]] = None,
    image_detail: str = "low",
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
) -> Tuple[str, float, dict[str, Any]]:
    """Generate text and capture usage/cost metadata in a single call."""
    if model is None:
        model = get_default_model()
    _, reasoning_value = _normalize_reasoning_level(reasoning_level)

    text, cost_usd, usage, _ = _perform_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_value,
    )
    return text, cost_usd, usage


# ---------------------------------------------------------------------------
# Convenience wrappers around the text helpers
# ---------------------------------------------------------------------------


def generate_text_gpt_with_cost(
    prompt: str,
    system_prompt: str = "",
    image: bytes | bytearray | str | None = None,
    multi_image: Optional[Sequence[bytes]] = None,
    image_detail: str = "low",
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
) -> Tuple[str, float, dict[str, Any]]:
    """Backward-compatible alias around `generate_text_with_cost`."""
    return generate_text_with_cost(
        prompt=prompt,
        system_prompt=system_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_level,
    )


def generate_text(
    prompt: str,
    system_prompt: str = "",
    image: bytes | bytearray | str | None = None,
    multi_image: Optional[Sequence[bytes]] = None,
    image_detail: str = "low",
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    model: str | None = None,
) -> str:
    """Convenience helper returning only the generated text."""
    return generate_text_gpt_with_cost(
        prompt=prompt,
        system_prompt=system_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_level,
    )[0]


def answer_question_with_vision(
    question: str,
    screenshot: Optional[Union[bytes, Sequence[bytes]]],
    *,
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
) -> Optional[bool]:
    """Use a vision-capable model to answer a yes/no question about screenshots."""
    if model is None:
        model = get_default_model()
    if reasoning_level is None:
        final_reasoning_level = ReasoningLevel.coerce(get_default_reasoning_level())
    else:
        final_reasoning_level = ReasoningLevel.coerce(reasoning_level)
    question = (question or "").strip()
    if not screenshot or not question:
        return None

    image: Union[bytes, bytearray, str, None]
    multi_image: Optional[Sequence[bytes]] = None

    if isinstance(screenshot, (bytes, bytearray)):
        image = screenshot
    elif isinstance(screenshot, SequenceABC) and not isinstance(screenshot, (str, bytes, bytearray)):
        screenshot_list = list(screenshot)
        if not screenshot_list:
            return None
        image = screenshot_list[0]
        if len(screenshot_list) > 1:
            multi_image = screenshot_list[1:]
    else:
        image = screenshot

    system_prompt = (
        "You are a careful web QA assistant. Look at the screenshot(s) and answer the question strictly using JSON.\n"
        "Reply with exactly one JSON object: {\"answer\": \"yes\"} or {\"answer\": \"no\"} (lowercase). No extra text."
    )
    prompt = (
        f"Question: {question}\n"
        "Respond with JSON only. Example: {\"answer\": \"yes\"}"
    )
    try:
        answer = (
            generate_text(
                prompt,
                system_prompt=system_prompt,
                image=image,
                multi_image=multi_image,
                model=model,
                reasoning_level=final_reasoning_level,
            )
            .strip()
        )
    except Exception:
        return None

    if not answer:
        return None
    cleaned = answer.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("` \n")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        data = json.loads(cleaned)
        value = str(data.get("answer", "")).strip().lower()
        if value in {"yes", "true"}:
            return True
        if value in {"no", "false"}:
            return False
    except Exception:
        pass

    lowered = cleaned.lower()
    if lowered.startswith("yes"):
        return True
    if lowered.startswith("no"):
        return False
    return None


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------


def generate_model_with_cost(
    prompt: str,
    model_object_type: Optional[Type[BaseModel]] = None,
    system_prompt: str = "",
    image: Union[bytes, bytearray, str, None] = None,
    multi_image: Optional[Sequence[bytes]] = None,
    image_detail: str = "low",
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
) -> Tuple[Any, float, dict[str, Any]]:
    """Generate structured output and capture usage/cost metadata."""
    if model is None:
        model = get_default_model()
    _, reasoning_value = _normalize_reasoning_level(reasoning_level)

    response_format = _build_response_format(model_object_type, model)
    
    # Add explicit JSON instructions to system prompt for all models when using structured output
    # This is a workaround for LiteLLM bug #16813 where models don't return proper JSON
    enhanced_system_prompt = system_prompt
    if model_object_type:
        json_instruction = "\n\nIMPORTANT: You MUST respond with valid JSON only. Do not include any markdown formatting, explanations, or text outside the JSON object."
        # Include schema for better guidance
        try:
            schema = model_object_type.model_json_schema()
        except AttributeError:
            schema = model_object_type.schema()
        json_instruction += f"\n\nExpected JSON schema:\n{json.dumps(schema, indent=2)}"
        enhanced_system_prompt = (system_prompt + json_instruction) if system_prompt else json_instruction.strip()

    text, cost_usd, usage, response = _perform_completion(
        prompt=prompt,
        system_prompt=enhanced_system_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_value,
        response_format=response_format,
    )

    if model_object_type is None:
        parsed_result: Any = text
    else:
        # Safely extract message from response
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            choices = [{}]
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        parsed_obj = message.get("parsed") if isinstance(message, dict) else None
        
        # Check if parsed_obj is valid - it should be a dict or a Pydantic model instance
        if parsed_obj is not None:
            # If it's already a Pydantic model instance, use it
            if isinstance(parsed_obj, model_object_type):
                parsed_result = parsed_obj
            # If it's a dict, validate it
            elif isinstance(parsed_obj, dict):
                try:
                    if hasattr(model_object_type, "model_validate"):
                        parsed_result = model_object_type.model_validate(parsed_obj)
                    else:
                        parsed_result = model_object_type(**parsed_obj)
                except ValidationError as exc:
                    print(f"⚠️ Pre-parsed object validation failed for {model_object_type.__name__}: {exc}")
                    # Fall back to manual parsing
                    manual = _manual_parse_structured_output(text, model_object_type)
                    parsed_result = manual if manual is not None else text
            # If it's a primitive type (bool, str, int, etc.), try to parse from text instead
            else:
                print(f"⚠️ LLM returned primitive type {type(parsed_obj).__name__} instead of object for {model_object_type.__name__}")
                manual = _manual_parse_structured_output(text, model_object_type)
                parsed_result = manual if manual is not None else text
        else:
            try:
                parsed_result = model_object_type.model_validate_json(text)  # type: ignore[attr-defined]
            except AttributeError:
                try:
                    parsed_result = model_object_type.parse_raw(text)  # type: ignore[attr-defined]
                except Exception as exc:
                    manual = _manual_parse_structured_output(text, model_object_type)
                    if manual is not None:
                        parsed_result = manual
                    else:
                        print(f"⚠️ Structured output parse failed (parse_raw) for {model_object_type.__name__}: {exc}")
                        parsed_result = text
            except (ValidationError, json.JSONDecodeError) as exc:
                manual = _manual_parse_structured_output(text, model_object_type)
                if manual is not None:
                    parsed_result = manual
                else:
                    print(f"⚠️ Structured output parse failed for {model_object_type.__name__}: {exc}")
                    parsed_result = text
            except Exception as exc:
                manual = _manual_parse_structured_output(text, model_object_type)
                if manual is not None:
                    parsed_result = manual
                else:
                    print(f"⚠️ Unexpected error parsing structured output for {model_object_type.__name__}: {exc}")
                    parsed_result = text

    return parsed_result, cost_usd, usage


def generate_model_gpt_with_cost(
    prompt: str,
    model_object_type: Optional[Type[BaseModel]] = None,
    system_prompt: str = "",
    image: Union[bytes, bytearray, str, None] = None,
    multi_image: Optional[Sequence[bytes]] = None,
    image_detail: str = "low",
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
) -> Tuple[Any, float, dict[str, Any]]:
    """Backward-compatible alias around `generate_model_with_cost`."""
    return generate_model_with_cost(
        prompt=prompt,
        model_object_type=model_object_type,
        system_prompt=system_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_level,
    )


def generate_model_gpt(
    prompt: str,
    model_object_type: Optional[Type[BaseModel]] = None,
    system_prompt: str = "",
    image: Union[bytes, bytearray, str, None] = None,
    multi_image: Optional[Sequence[bytes]] = None,
    image_detail: str = "low",
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    model: str | None = None,
) -> Any:
    """Convenience helper returning only the parsed structured output."""
    return generate_model_gpt_with_cost(
        prompt,
        model_object_type=model_object_type,
        system_prompt=system_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_level,
    )[0]


def generate_model(
    prompt: str,
    model_object_type: Optional[Type[BaseModel]] = None,
    system_prompt: str = "",
    image: Union[bytes, bytearray, str, None] = None,
    multi_image: Optional[Sequence[bytes]] = None,
    image_detail: str = "low",
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    model: str | None = None,
) -> Any:
    """Public entry-point mirroring previous API semantics."""
    return generate_model_gpt(
        prompt=prompt,
        model_object_type=model_object_type,
        system_prompt=system_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        reasoning_level=reasoning_level,
        model=model,
    )

