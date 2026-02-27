"""Shared utilities for invoking OpenAI chat/vision models.

This module provides a consistent way to interact with OpenAI models using the Responses API.

Features:
- Native structured output support (Pydantic models)
- Streaming support for both text and structured output
- Vision support (single and multiple images)
- Cost tracking and token usage
- Fallback manual parsing for edge cases
"""

from __future__ import annotations

import base64
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from utils.debug_print import dprint, get_print_mode, PrintMode

# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================


class ReasoningLevel(str, Enum):
    """Supported reasoning effort levels for providers that expose them."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def coerce(cls, value: Union["ReasoningLevel", str]) -> "ReasoningLevel":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(level.value for level in cls)
            raise ValueError(
                f"Invalid reasoning level '{value}'. Allowed values: {allowed}."
            ) from exc


@dataclass
class ProviderResponse:
    """Standardized response from any provider."""

    text: str
    parsed: Optional[BaseModel]  # Pre-parsed object from provider (if structured)
    usage: dict[str, int]  # input_tokens, output_tokens, total_tokens
    cost_usd: float
    raw_response: Any


@dataclass
class StreamChunk:
    """A single chunk in a streaming response."""

    delta: str  # Incremental text
    is_final: bool  # Whether this is the last chunk
    parsed: Optional[BaseModel] = None  # Parsed object in final chunk (structured)
    usage: Optional[dict] = None  # Only present in final chunk


class UnsupportedModelError(Exception):
    """Raised when a model provider is not implemented."""

    pass


class ProviderAPIError(Exception):
    """Raised when provider API returns an error."""

    pass


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Global defaults – kept in module state so callers can change them centrally.
_DEFAULT_MODEL = "gpt-5-mini"
_DEFAULT_REASONING_LEVEL: str = ReasoningLevel.MEDIUM.value
_DEFAULT_AGENT_MODEL = "gpt-5-mini"
_DEFAULT_AGENT_REASONING_LEVEL: str = ReasoningLevel.MEDIUM.value
_DEFAULT_TEMPERATURE: Optional[float] = None  # None means use provider default

# Optional per-model or per-provider API keys. Populate this if you do not want
# to rely solely on environment variables.
MODEL_API_KEYS: dict[str, str] = {}

# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================


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


def set_default_temperature(temperature: Optional[float]) -> None:
    """Set the default temperature used when callers omit the parameter."""
    global _DEFAULT_TEMPERATURE
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("Temperature must be between 0.0 and 2.0")
    _DEFAULT_TEMPERATURE = temperature


def get_default_temperature() -> Optional[float]:
    """Return the globally configured default temperature."""
    return _DEFAULT_TEMPERATURE


def _is_debug_mode() -> bool:
    """Check if debug mode is enabled (uses global PrintMode from debug_print)."""
    return get_print_mode() == PrintMode.DEBUG


# ============================================================================
# ABSTRACT BASE PROVIDER
# ============================================================================


class ModelProvider(ABC):
    """Base class for all model providers."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        model: str,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> Union[ProviderResponse, Iterator[StreamChunk]]:
        """
        Generate completion (text or structured).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier
            reasoning_effort: Reasoning level string if supported
            response_format: Pydantic BaseModel class for structured output
            temperature: Sampling temperature (0.0 to 2.0), None for provider default
            stream: Whether to stream the response

        Returns:
            ProviderResponse if stream=False
            Iterator[StreamChunk] if stream=True
        """
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider supports image inputs."""
        pass

    @abstractmethod
    def supports_reasoning(self, model: str) -> bool:
        """Whether this model supports reasoning parameters."""
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming."""
        pass

    @abstractmethod
    def calculate_cost(self, usage: dict, model: str) -> float:
        """Calculate cost in USD based on usage and model."""
        pass


# ============================================================================
# OPENAI PROVIDER (Responses API)
# ============================================================================


class OpenAIProvider(ModelProvider):
    """OpenAI implementation using Responses API with native structured outputs."""

    def __init__(self):
        try:
            from openai import OpenAI
            api_key = self._get_api_key()
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI Python SDK is required for OpenAI models. "
                "Install with: pip install openai"
            )

    def complete(
        self,
        messages: list[dict],
        model: str,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> Union[ProviderResponse, Iterator[StreamChunk]]:

        # Convert messages to OpenAI Responses API format
        input_messages = self._convert_to_openai_format(messages)

        # Build optional parameters
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature

        # STRUCTURED OUTPUT PATH - responses.parse()
        if response_format:
            try:
                response = self.client.responses.parse(
                    model=model,
                    input=input_messages,
                    text_format=response_format,
                    stream=stream,
                    **kwargs,
                )

                if stream:
                    return self._handle_structured_stream(response, model)
                else:
                    # Debug logging
                    # if _is_debug_mode():
                    #     dprint(f"🔍 OpenAI Response (structured): {response}")

                    parsed_obj = response.output_parsed

                    # Check if parsed object is None or empty
                    if parsed_obj is None:
                        dprint("⚠️ OpenAI returned None for output_parsed")
                        dprint(f"Full response: {response}")
                        raise ProviderAPIError(
                            f"OpenAI returned empty parsed output. "
                            f"Response: {response}"
                        )

                    usage = self._extract_usage(response)
                    cost = self.calculate_cost(usage, model)

                    return ProviderResponse(
                        text=parsed_obj.model_dump_json(),
                        parsed=parsed_obj,
                        usage=usage,
                        cost_usd=cost,
                        raw_response=response,
                    )
            except Exception as e:
                dprint(f"❌ OpenAI structured output error: {e}")
                dprint(f"Model: {model}")
                dprint(f"Input messages count: {len(input_messages)}")
                raise ProviderAPIError(f"OpenAI structured output failed: {e}")

        # TEXT GENERATION PATH - responses.create()
        else:
            try:
                response = self.client.responses.create(
                    model=model,
                    input=input_messages,
                    stream=stream,
                    **kwargs,
                )

                if stream:
                    return self._handle_text_stream(response, model)
                else:

                    text = response.output_text

                    # Check if text is None or empty
                    if text is None or text == "":
                        dprint("⚠️ OpenAI returned empty output_text")
                        dprint(f"Full response: {response}")
                        dprint(f"Response attributes: {dir(response)}")

                        # Check for refusal or other status
                        if hasattr(response, 'refusal') and response.refusal:
                            raise ProviderAPIError(
                                f"OpenAI refused the request: {response.refusal}"
                            )

                        if hasattr(response, 'status') and response.status:
                            raise ProviderAPIError(
                                f"OpenAI response status: {response.status}"
                            )

                        raise ProviderAPIError(
                            f"OpenAI returned empty output. "
                            f"Response: {response}"
                        )

                    usage = self._extract_usage(response)
                    cost = self.calculate_cost(usage, model)

                    return ProviderResponse(
                        text=text,
                        parsed=None,
                        usage=usage,
                        cost_usd=cost,
                        raw_response=response,
                    )
            except Exception as e:
                dprint(f"❌ OpenAI text generation error: {e}")
                dprint(f"Model: {model}")
                dprint(f"Input messages count: {len(input_messages)}")
                raise ProviderAPIError(f"OpenAI text generation failed: {e}")

    def _convert_to_openai_format(self, messages: list[dict]) -> list[dict]:
        """Convert provider-agnostic messages to OpenAI Responses API format."""
        openai_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Handle multi-part content (text + images)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "input_text":
                        parts.append({
                            "type": "input_text",
                            "text": part["text"]
                        })
                    elif part.get("type") == "input_image":
                        image_data = part.get("image_url", {})
                        parts.append({
                            "type": "input_image",
                            "image_url": image_data.get("url"),
                            "detail": image_data.get("detail", "high")
                        })

                openai_messages.append({
                    "role": role,
                    "content": parts
                })

            # Handle simple text content
            else:
                openai_messages.append({
                    "role": role,
                    "content": content
                })

        return openai_messages

    def _handle_structured_stream(
        self, stream_response, model: str
    ) -> Iterator[StreamChunk]:
        """Handle streaming structured output."""
        for chunk in stream_response:
            delta = getattr(chunk, "output_text_delta", "")
            is_final = hasattr(chunk, "output_parsed") and chunk.output_parsed is not None

            if is_final:
                usage = self._extract_usage(chunk) if hasattr(chunk, "usage") else None
                yield StreamChunk(
                    delta=delta,
                    is_final=True,
                    parsed=chunk.output_parsed,
                    usage=usage,
                )
            else:
                yield StreamChunk(
                    delta=delta,
                    is_final=False,
                    parsed=None,
                    usage=None,
                )

    def _handle_text_stream(self, stream_response, model: str) -> Iterator[StreamChunk]:
        """Handle streaming text generation."""
        for chunk in stream_response:
            delta = getattr(chunk, "output_text_delta", "")
            is_final = hasattr(chunk, "finish_reason") and chunk.finish_reason is not None

            usage = None
            if is_final and hasattr(chunk, "usage"):
                usage = self._extract_usage(chunk)

            yield StreamChunk(
                delta=delta,
                is_final=is_final,
                parsed=None,
                usage=usage,
            )

    def _extract_usage(self, response) -> dict:
        """Extract token usage from OpenAI response."""
        if hasattr(response, "usage"):
            usage = response.usage
            return {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment or MODEL_API_KEYS."""
        return MODEL_API_KEYS.get("openai") or os.getenv("OPENAI_API_KEY") or ""

    def supports_vision(self) -> bool:
        return True

    def supports_reasoning(self, model: str) -> bool:
        model_lower = model.lower()
        return "gpt-4o" in model_lower or "o1" in model_lower or "gpt-5" in model_lower

    def supports_streaming(self) -> bool:
        return True

    def calculate_cost(self, usage: dict, model: str) -> float:
        """Calculate cost based on OpenAI pricing."""
        pricing = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-5-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-5": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        }

        model_lower = model.lower()
        for key, price in pricing.items():
            if key in model_lower:
                return (
                    usage.get("input_tokens", 0) * price["input"]
                    + usage.get("output_tokens", 0) * price["output"]
                )

        return 0.0




# ============================================================================
# PROVIDER REGISTRY
# ============================================================================


def get_provider(model: str) -> ModelProvider:
    """
    Get the OpenAI provider instance.

    Args:
        model: Model identifier (e.g., "gpt-4o", "gpt-5-mini")

    Returns:
        OpenAIProvider instance
    """
    return OpenAIProvider()


# ============================================================================
# SHARED UTILITIES
# ============================================================================


def _detect_image_type(image_bytes: bytes) -> str:
    """Determine MIME suffix for raw image bytes."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"
    return "jpeg"


def _prepare_image_part(
    image: Union[bytes, bytearray, str], image_detail: str
) -> dict:
    """Convert image to provider-agnostic format."""

    if isinstance(image, (bytes, bytearray)):
        # Detect image type and encode as base64
        detected_type = _detect_image_type(image)
        b64 = base64.b64encode(image).decode("ascii")
        url = f"data:image/{detected_type};base64,{b64}"
    else:
        # Already a string (base64 or URL)
        url = str(image)
        if not url.startswith("data:image/") and not url.startswith("http"):
            url = f"data:image/jpeg;base64,{url}"

    return {
        "type": "input_image",
        "image_url": {"url": url, "detail": image_detail},
    }


def _build_messages(
    system_prompt: str,
    developer_prompt: str,
    prompt: str,
    image: Optional[Union[bytes, bytearray, str]] = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
) -> list[dict]:
    """
    Build messages with support for multiple images.

    Returns messages in a provider-agnostic format that each provider
    will convert to their specific format.
    """
    messages = []

    # System message
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if developer_prompt:
        messages.append({"role": "developer", "content": developer_prompt})

    # User message with text and images
    content_parts = []

    # Add text
    if prompt:
        content_parts.append({"type": "input_text", "text": prompt})

    # Add images
    all_images = []
    if multi_image:
        all_images.extend(multi_image)
    if image is not None:
        all_images.append(image)

    for img in all_images:
        image_part = _prepare_image_part(img, image_detail)
        content_parts.append(image_part)

    # Use multi-part content if we have images, otherwise simple text
    if len(content_parts) > 1 or all_images:
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": prompt})

    return messages


def _extract_json_object(text: str) -> Optional[Any]:
    """Best-effort extraction of a JSON object embedded in arbitrary text."""
    if not isinstance(text, str):
        return None

    candidates: List[str] = []
    stripped = text.strip()

    # Extract from markdown code blocks
    if stripped.startswith("```"):
        for block in re.findall(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL
        ):
            candidates.append(block)

    candidates.append(stripped)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate_stripped = candidate.strip()

        # Remove control characters
        candidate_cleaned = re.sub(
            r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", " ", candidate_stripped
        )

        # Remove trailing commas
        candidate_cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate_cleaned)

        # Replace literal newlines with spaces
        candidate_cleaned = candidate_cleaned.replace("\n", " ").replace("\r", " ")

        try:
            return json.loads(candidate_cleaned)
        except Exception:
            pass

        # Try to find JSON object starting with {
        for idx, ch in enumerate(candidate_cleaned):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(candidate_cleaned[idx:])
                    return obj
                except Exception:
                    continue

    return None


def _manual_parse_structured_output(
    text: str, model_object_type: Type[BaseModel]
) -> Optional[BaseModel]:
    """Attempt lightweight manual parsing when the model fails to return valid JSON."""
    try:
        # Try to parse simple yes/no boolean responses
        cleaned = text.strip()

        # Match "field_name: value" patterns
        simple_bool_match = re.match(
            r"^(\w+)\s*[:(]?\s*(true|false|yes|no)\s*[)]?$", cleaned, re.IGNORECASE
        )
        if simple_bool_match:
            field_name = simple_bool_match.group(1)
            value_str = simple_bool_match.group(2).lower()
            bool_value = value_str in ("true", "yes")
            try:
                if hasattr(model_object_type, "model_validate"):
                    return model_object_type.model_validate({field_name: bool_value})
                return model_object_type(**{field_name: bool_value})
            except ValidationError as exc:
                dprint(
                    f"⚠️ Simple boolean parse failed validation for {model_object_type.__name__}: {exc}"
                )

        # Handle plain boolean text
        if cleaned.lower() in ("true", "false", "yes", "no"):
            bool_value = cleaned.lower() in ("true", "yes")
            try:
                schema = (
                    model_object_type.model_json_schema()
                    if hasattr(model_object_type, "model_json_schema")
                    else model_object_type.schema()
                )
                properties = schema.get("properties", {})
                bool_fields = [
                    name
                    for name, prop in properties.items()
                    if prop.get("type") == "boolean"
                ]
                if len(bool_fields) == 1:
                    field_name = bool_fields[0]
                    if hasattr(model_object_type, "model_validate"):
                        return model_object_type.model_validate(
                            {field_name: bool_value}
                        )
                    return model_object_type(**{field_name: bool_value})
            except Exception as exc:
                dprint(
                    f"⚠️ Plain boolean inference failed for {model_object_type.__name__}: {exc}"
                )

        # Generic fallback: try to extract JSON
        json_obj = _extract_json_object(text)
        if isinstance(json_obj, dict):
            try:
                if hasattr(model_object_type, "model_validate"):
                    return model_object_type.model_validate(json_obj)
                return model_object_type(**json_obj)
            except ValidationError as exc:
                dprint(
                    f"⚠️ Generic JSON extraction failed validation for {model_object_type.__name__}: {exc}"
                )
                return None

    except Exception as exc:
        dprint(f"⚠️ Manual parse helper error for {model_object_type.__name__}: {exc}")

    return None


def _normalize_reasoning_level(
    reasoning_level: Union[ReasoningLevel, str, None]
) -> tuple[ReasoningLevel, Optional[str]]:
    """Normalize caller input to (enum, provider_value)."""
    if reasoning_level is None:
        reasoning_enum = ReasoningLevel.coerce(get_default_reasoning_level())
    else:
        reasoning_enum = ReasoningLevel.coerce(reasoning_level)

    if reasoning_enum is ReasoningLevel.NONE:
        return reasoning_enum, None
    return reasoning_enum, reasoning_enum.value


# ============================================================================
# PUBLIC API - NON-STREAMING
# ============================================================================


def generate_text_with_cost(
    prompt: str,
    system_prompt: str = "",
    developer_prompt: str = "",
    image: bytes | bytearray | str | None = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    temperature: Optional[float] = None,
) -> Tuple[str, float, dict[str, Any]]:
    """Generate text and capture usage/cost metadata in a single call."""
    if model is None:
        model = get_default_model()

    _, reasoning_value = _normalize_reasoning_level(reasoning_level)

    if temperature is None:
        temperature = get_default_temperature()

    provider = get_provider(model)
    messages = _build_messages(
        system_prompt, developer_prompt, prompt, image, multi_image, image_detail
    )

    response = provider.complete(
        messages=messages,
        model=model,
        reasoning_effort=reasoning_value,
        temperature=temperature,
        stream=False,
    )

    # Log cost
    try:
        from utils.event_logger import get_event_logger

        get_event_logger().llm_cost(
            cost_usd=response.cost_usd,
            input_tokens=response.usage["input_tokens"],
            output_tokens=response.usage["output_tokens"],
            total_tokens=response.usage["total_tokens"],
            model=model,
        )
    except Exception:
        pass

    return response.text, response.cost_usd, response.usage


def generate_text_gpt_with_cost(
    prompt: str,
    system_prompt: str = "",
    developer_prompt: str = "",
    image: bytes | bytearray | str | None = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    temperature: Optional[float] = None,
) -> Tuple[str, float, dict[str, Any]]:
    """Backward-compatible alias around generate_text_with_cost."""
    return generate_text_with_cost(
        prompt=prompt,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_level,
        temperature=temperature,
    )


def generate_text(
    prompt: str,
    system_prompt: str = "",
    developer_prompt: str = "",
    image: bytes | bytearray | str | None = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    model: str | None = None,
    temperature: Optional[float] = None,
) -> str:
    """Convenience helper returning only the generated text."""
    return generate_text_gpt_with_cost(
        prompt=prompt,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_level,
        temperature=temperature,
    )[0]


def generate_model_with_cost(
    prompt: str,
    model_object_type: Optional[Type[BaseModel]] = None,
    system_prompt: str = "",
    developer_prompt: str = "",
    image: Union[bytes, bytearray, str, None] = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    temperature: Optional[float] = None,
) -> Tuple[Any, float, dict[str, Any]]:
    """Generate structured output and capture usage/cost metadata."""
    if model is None:
        model = get_default_model()

    _, reasoning_value = _normalize_reasoning_level(reasoning_level)

    if temperature is None:
        temperature = get_default_temperature()

    provider = get_provider(model)
    messages = _build_messages(
        system_prompt, developer_prompt, prompt, image, multi_image, image_detail
    )

    response = provider.complete(
        messages=messages,
        model=model,
        reasoning_effort=reasoning_value,
        response_format=model_object_type,
        temperature=temperature,
        stream=False,
    )

    # Log cost
    try:
        from utils.event_logger import get_event_logger

        get_event_logger().llm_cost(
            cost_usd=response.cost_usd,
            input_tokens=response.usage["input_tokens"],
            output_tokens=response.usage["output_tokens"],
            total_tokens=response.usage["total_tokens"],
            model=model,
        )
    except Exception:
        pass

    # Return parsed object or fallback to text
    if response.parsed is not None:
        return response.parsed, response.cost_usd, response.usage
    else:
        # Fallback parsing
        if model_object_type:
            try:
                parsed = model_object_type.model_validate_json(response.text)
                return parsed, response.cost_usd, response.usage
            except Exception:
                parsed = _manual_parse_structured_output(response.text, model_object_type)
                if parsed is not None:
                    return parsed, response.cost_usd, response.usage

        # Return raw text if all parsing fails
        return response.text, response.cost_usd, response.usage


def generate_model_gpt_with_cost(
    prompt: str,
    model_object_type: Optional[Type[BaseModel]] = None,
    system_prompt: str = "",
    developer_prompt: str = "",
    image: Union[bytes, bytearray, str, None] = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    temperature: Optional[float] = None,
) -> Tuple[Any, float, dict[str, Any]]:
    """Backward-compatible alias around generate_model_with_cost."""
    return generate_model_with_cost(
        prompt=prompt,
        model_object_type=model_object_type,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_level,
        temperature=temperature,
    )


def generate_model_gpt(
    prompt: str,
    model_object_type: Optional[Type[BaseModel]] = None,
    system_prompt: str = "",
    developer_prompt: str = "",
    image: Union[bytes, bytearray, str, None] = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    model: str | None = None,
    temperature: Optional[float] = None,
) -> Any:
    """Convenience helper returning only the parsed structured output."""
    return generate_model_gpt_with_cost(
        prompt,
        model_object_type=model_object_type,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        model=model,
        reasoning_level=reasoning_level,
        temperature=temperature,
    )[0]


def generate_model(
    prompt: str,
    model_object_type: Optional[Type[BaseModel]] = None,
    system_prompt: str = "",
    developer_prompt: str = "",
    image: Union[bytes, bytearray, str, None] = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    model: str | None = None,
    temperature: Optional[float] = None,
) -> Any:
    """Public entry-point mirroring previous API semantics."""
    return generate_model_gpt(
        prompt=prompt,
        model_object_type=model_object_type,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        image=image,
        multi_image=multi_image,
        image_detail=image_detail,
        reasoning_level=reasoning_level,
        model=model,
        temperature=temperature,
    )


def answer_question_with_vision(
    question: str,
    screenshot: Optional[Union[bytes, list[bytes]]],
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
    multi_image: Optional[list[bytes]] = None

    if isinstance(screenshot, (bytes, bytearray)):
        image = screenshot
    elif isinstance(screenshot, Sequence) and not isinstance(
        screenshot, (str, bytes, bytearray)
    ):
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
        'Reply with exactly one JSON object: {"answer": "yes"} or {"answer": "no"} (lowercase). No extra text.'
    )
    prompt = f"Question: {question}\n" 'Respond with JSON only. Example: {"answer": "yes"}'

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


# ============================================================================
# PUBLIC API - STREAMING
# ============================================================================


def generate_text_stream(
    prompt: str,
    system_prompt: str = "",
    developer_prompt: str = "",
    image: Optional[bytes] = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    model: Optional[str] = None,
    reasoning_level: Optional[str] = None,
) -> Iterator[StreamChunk]:
    """Generate text with streaming."""
    if model is None:
        model = get_default_model()

    _, reasoning_value = _normalize_reasoning_level(reasoning_level)

    provider = get_provider(model)
    messages = _build_messages(
        system_prompt, developer_prompt, prompt, image, multi_image, image_detail
    )

    return provider.complete(
        messages=messages,
        model=model,
        reasoning_effort=reasoning_value,
        stream=True,
    )


def generate_model_stream(
    prompt: str,
    model_object_type: Type[BaseModel],
    system_prompt: str = "",
    developer_prompt: str = "",
    image: Optional[bytes] = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    model: Optional[str] = None,
    reasoning_level: Optional[str] = None,
) -> Iterator[Union[StreamChunk, BaseModel]]:
    """
    Generate structured output with streaming.

    Yields:
        StreamChunk objects with incremental JSON/text
        Final yield is the fully parsed BaseModel instance (if successful)
    """
    if model is None:
        model = get_default_model()

    _, reasoning_value = _normalize_reasoning_level(reasoning_level)

    provider = get_provider(model)
    messages = _build_messages(
        system_prompt, developer_prompt, prompt, image, multi_image, image_detail
    )

    for chunk in provider.complete(
        messages=messages,
        model=model,
        reasoning_effort=reasoning_value,
        response_format=model_object_type,
        stream=True,
    ):
        if chunk.is_final and chunk.parsed:
            # Yield the final parsed object
            yield chunk.parsed
        else:
            # Yield intermediate chunks
            yield chunk


# ============================================================================
# PUBLIC API - FUNCTION CALLING
# ============================================================================


def generate_action_with_tools(
    prompt: str,
    tools: list[dict],
    system_prompt: str = "",
    developer_prompt: str = "",
    image: bytes | bytearray | str | None = None,
    multi_image: Optional[list[bytes]] = None,
    image_detail: str = "high",
    model: str | None = None,
    reasoning_level: Union[ReasoningLevel, str, None] = None,
    temperature: Optional[float] = None,
    tool_choice: str = "required",
    parallel_tool_calls: bool = True,
    previous_response_id: Optional[str] = None,
    tool_call_outputs: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Generate action using OpenAI function calling.

    Args:
        prompt: The prompt for the model
        tools: List of function tool definitions (OpenAI format)
        system_prompt: System prompt
        image: Optional image input
        multi_image: Optional multiple images
        image_detail: Image detail level
        model: Model to use (defaults to agent model)
        reasoning_level: Reasoning level
        temperature: Temperature setting
        tool_choice: "auto", "required", or "none"

    Returns:
        dict with:
        - 'function_name': name of function to call
        - 'arguments': dict of function arguments
        - 'usage': token usage dict
        - 'cost_usd': cost in USD
    """
    if model is None:
        model = get_default_agent_model()

    _, reasoning_value = _normalize_reasoning_level(reasoning_level)

    if temperature is None:
        temperature = get_default_temperature()

    # Build messages
    messages = _build_messages(
        system_prompt, developer_prompt, prompt, image, multi_image, image_detail
    )

    # Get OpenAI client
    provider = get_provider(model)
    if not isinstance(provider, OpenAIProvider):
        raise UnsupportedModelError(
            "Function calling is only supported with OpenAI models"
        )

    # Convert messages to OpenAI Responses input format
    openai_messages = provider._convert_to_openai_format(messages)

    # Convert Chat-Completions tool schema to Responses schema.
    responses_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            responses_tools.append(
                {
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object"}),
                }
            )
        elif tool.get("type") == "function" and "name" in tool:
            responses_tools.append(tool)

    # Call OpenAI Responses API with tools.
    try:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature

        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
            # Inject tool outputs from the previous function calls before new user input.
            # The Responses API requires outputs for all function calls in the previous response.
            if tool_call_outputs:
                openai_messages = list(tool_call_outputs) + openai_messages

        response = provider.client.responses.create(
            model=model,
            input=openai_messages,
            tools=responses_tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        # Extract usage and calculate cost
        usage = provider._extract_usage(response)
        cost = provider.calculate_cost(usage, model)

        # Log cost
        try:
            from utils.event_logger import get_event_logger
            get_event_logger().llm_cost(
                cost_usd=cost,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                model=model,
            )
        except Exception:
            pass

        # Extract tool calls from Responses output items.
        actions = []
        output_items = getattr(response, "output", None) or []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")
            if item_type not in {"function_call", "tool_call"}:
                continue

            fn_name = getattr(item, "name", None)
            if fn_name is None and isinstance(item, dict):
                fn_name = item.get("name")
            if not fn_name:
                continue

            raw_args = getattr(item, "arguments", None)
            if raw_args is None and isinstance(item, dict):
                raw_args = item.get("arguments")

            # Extract call_id (call_xxx format) for submitting tool outputs in the next request.
            # The Responses API has two ID fields: item.id (fc_xxx, item-level) and
            # item.call_id (call_xxx, required for function_call_output). Prioritize call_id.
            call_id = getattr(item, "call_id", None)
            if call_id is None and isinstance(item, dict):
                call_id = item.get("call_id") or item.get("id")

            parsed_args: Any = {}
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except Exception:
                    parsed_args = {}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args

            actions.append(
                {
                    "function_name": fn_name,
                    "arguments": parsed_args,
                    "usage": usage,
                    "cost_usd": cost,
                    "call_id": call_id,
                }
            )

        if not actions:
            if tool_choice == "required":
                raise ProviderAPIError("Model did not call any function despite tool_choice='required'")
            return [{
                "function_name": None,
                "arguments": None,
                "reasoning": getattr(response, "output_text", "") or "",
                "usage": usage,
                "cost_usd": cost,
                "response_id": getattr(response, "id", None),
            }]

        # Tag first action with response_id so callers can chain requests
        response_id = getattr(response, "id", None)
        if response_id:
            actions[0]["response_id"] = response_id

        return actions

    except Exception as e:
        dprint(f"❌ Function calling error: {e}")
        raise ProviderAPIError(f"Function calling failed: {e}")
