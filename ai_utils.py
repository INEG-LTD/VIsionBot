from typing import Any, Optional, Tuple, Type, Union
from google import genai
from openai import OpenAI

from pydantic import BaseModel

client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
client_gpt = OpenAI(api_key="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")

def generate_text_gemini(prompt: str, system_prompt: str = "", image = None, multi_image = None, model: str = "gemini-2.5-flash") -> str:
    contents = [prompt]
    if image:
        # Handle both bytes and genai.types.Part objects
        if isinstance(image, bytes):
            image = genai.types.Part.from_bytes(data=image, mime_type="image/png")
        contents.append(image)
    if multi_image and len(multi_image) > 0:
        for image in multi_image:
            if isinstance(image, bytes):
                image = genai.types.Part.from_bytes(data=image, mime_type="image/png")
            contents.append(image)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )
    return response.text

def generate_text_gpt_with_cost(
    prompt: str,
    system_prompt: str = "",
    image: bytes | str | None = None,
    multi_image: list[bytes] = None,
    image_detail: str = "low",
    model: str = "gpt-5-mini",
    reasoning_level: str = "low",
):
    """
    Returns (output_text, cost_usd, usage_dict).

    Notes:
    - Uses Responses API `usage` to compute cost precisely.
    - If `image` is bytes, we base64-encode it. If it's already a base64 str, we use it as-is.
    - `system_prompt` is sent as a top-level developer message (correct structure).
    """

    import base64

    messages = []
    if system_prompt:
        messages.append({"role": "developer", "content": system_prompt})

    content_parts = [{"type": "input_text", "text": prompt}]

    if multi_image and len(multi_image) > 0:
        for image in multi_image:
            b64 = base64.b64encode(image).decode("ascii")
            data_url = f"data:image/jpeg;base64,{b64}"
            content_parts.append({"type": "input_image", "image_url": data_url, "detail": image_detail})

    if image is not None:
        if isinstance(image, (bytes, bytearray)):
            b64 = base64.b64encode(image).decode("ascii")
        else:
            # assume base64 string (without data: header)
            b64 = str(image)
        content_parts.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}",
            "detail": image_detail
        })

    messages.append({"role": "user", "content": content_parts})

    response = client_gpt.responses.create(
        model=model,
        input=messages,
        reasoning={'effort': reasoning_level}
    )

    # ---- Cost calculation (USD) --------------------------------------------
    # Official per-1M-token prices
    PRICE_PER_MILLION = {
        # Source: OpenAI announcement / model docs
        # GPT-4o mini: $0.15 input / $0.60 output per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},  #  [oai_citation:1‡OpenAI Platform](https://platform.openai.com/docs/models/chatgpt-4o-latest?utm_source=chatgpt.com)
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
    }

    usage = getattr(response, "usage", None)
    # Responses API exposes usage as attributes or dict-like; handle both
    def _get(u, key):
        if u is None:
            return None
        return getattr(u, key, None) if not isinstance(u, dict) else u.get(key)

    input_tokens = _get(usage, "input_tokens") or 0
    output_tokens = _get(usage, "output_tokens") or 0

    price = PRICE_PER_MILLION.get(model)
    if price is None:
        # Unknown model: don’t guess — report zero and let caller decide.
        cost_usd = 0.0
    else:
        cost_usd = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    output_text = getattr(response, "output_text", None)
    # Fallback in case SDK version doesn’t expose output_text:
    if output_text is None:
        try:
            # Typical Responses API structure; be defensive
            output_text = response.output[0].content[0].text
        except Exception:
            output_text = ""

    print(f"Prompt Cost: {cost_usd} USD, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Tokens: {_get(usage, 'total_tokens')}")

    # Return the text AND the cost, plus raw usage for transparency
    return output_text, cost_usd, {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _get(usage, "total_tokens"),
    }

def generate_text_gpt(prompt: str, system_prompt: str = "", image: bytes = None, multi_image: list[bytes] = None, image_detail: str = "low", reasoning_level: str = "minimal", model: str = "gpt-5-mini") -> str:
    return generate_text_gpt_with_cost(
        prompt=prompt, 
        system_prompt=system_prompt, 
        image=image, 
        multi_image=multi_image, 
        image_detail=image_detail, 
        model=model, reasoning_level=reasoning_level)[0]

def generate_text(prompt: str, system_prompt: str = "", image: bytes = None, multi_image: list[bytes] = None, image_detail: str = "low", reasoning_level: str = "minimal", model: str = "gpt-5-mini") -> str:
    if model.startswith("gpt"):
        return generate_text_gpt(
            prompt=prompt, 
            system_prompt=system_prompt, 
            image=image, 
            multi_image=multi_image, 
            image_detail=image_detail, 
            reasoning_level=reasoning_level, model=model)
    elif model.startswith("gemini"):
        return generate_text_gemini(
            prompt=prompt, 
            system_prompt=system_prompt, 
            image=image, 
            multi_image=multi_image, 
            model=model)
    else:
        raise ValueError(f"Invalid model: {model}")

def generate_model_gpt_with_cost(
    prompt: str,
    model_object_type: Optional[Type[BaseModel]] = None,
    system_prompt: str = "",
    image: Union[bytes, str, None] = None,
    multi_image: list[bytes] = None,
    image_detail: str = "low",
    model: str = "gpt-5-mini",
    reasoning_level: str = "low",
) -> Tuple[Any, float, dict]:
    """
    Returns (parsed_result, cost_usd, usage_dict).

    - Sends `system_prompt` as a top-level developer message (correct schema).
    - Accepts image as bytes (base64-encodes) or prebase64 string (or data: URL).
    - Computes cost from response.usage using official per-1M token prices.
    - If `model_object_type` is None, falls back to text output.
    """

    import base64

    # ---- Build messages (correct Roles schema) ------------------------------
    messages = []
    if system_prompt:
        messages.append({"role": "developer", "content": system_prompt})

    parts = [{"type": "input_text", "text": prompt}]

    if multi_image and len(multi_image) > 0:
        for image in multi_image:
            b64 = base64.b64encode(image).decode("ascii")
            data_url = f"data:image/jpeg;base64,{b64}"
            parts.append({"type": "input_image", "image_url": data_url, "detail": image_detail})

    if image is not None:
        if isinstance(image, (bytes, bytearray)):
            b64 = base64.b64encode(image).decode("ascii")
            data_url = f"data:image/jpeg;base64,{b64}"
        else:
            s = str(image)
            data_url = s if s.startswith("data:image/") else f"data:image/jpeg;base64,{s}"
        parts.append({"type": "input_image", "image_url": data_url, "detail": image_detail})

    messages.append({"role": "user", "content": parts})

    # ---- Call API: parse if schema provided, else create --------------------
    if model_object_type is not None:
        response = client_gpt.responses.parse(
            model=model,
            input=messages,
            text_format=model_object_type,
            reasoning={'effort': reasoning_level}
        )
    else:
        response = client_gpt.responses.create(
            model=model,
            input=messages,
            reasoning={'effort': reasoning_level}
        )

    # ---- Usage & Cost -------------------------------------------------------
    PRICE_PER_MILLION = {
        # USD per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        # Add any other models you use here.
    }

    usage = getattr(response, "usage", None)
    def _get(u, key):
        if u is None:
            return 0
        return (getattr(u, key, 0) if not isinstance(u, dict) else u.get(key, 0)) or 0

    input_tokens = _get(usage, "input_tokens")
    output_tokens = _get(usage, "output_tokens")

    price = PRICE_PER_MILLION.get(model)
    cost_usd = 0.0 if price is None else (
        (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    )

    # ---- Extract parsed result (robust to SDK variants) --------------------
    parsed = getattr(response, "output_parsed", None)

    if parsed is None:
        # Fall back to text, and if a schema exists, try to validate locally.
        text = getattr(response, "output_text", None)
        if text is None:
            try:
                text = response.output[0].content[0].text
            except Exception:
                text = ""

        if model_object_type is not None:
            try:
                # Pydantic v2
                parsed = model_object_type.model_validate_json(text)
            except AttributeError:
                try:
                    # Pydantic v1
                    parsed = model_object_type.parse_raw(text)
                except Exception:
                    parsed = text
        else:
            parsed = text

    print(f"Prompt Cost: {cost_usd} USD, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Tokens: {_get(usage, 'total_tokens')}")

    return parsed, cost_usd, {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _get(usage, "total_tokens"),
    }

def generate_model_gemini(prompt: str, model_object_type: Optional[Type[BaseModel]] = None, system_prompt: str = "", image = None, multi_image = None, thinking_level: str = "none", model: str = "gemini-2.5-flash") -> BaseModel:
    contents = []
    if image:
        # Handle both bytes and genai.types.Part objects
        if isinstance(image, bytes):
            image = genai.types.Part.from_bytes(data=image, mime_type="image/png")
        contents.append(image)
    if multi_image and len(multi_image) > 0:
        for image in multi_image:
            if isinstance(image, bytes):
                image = genai.types.Part.from_bytes(data=image, mime_type="image/png")
            contents.append(image)
    contents.append(prompt)
    thinking_config=genai.types.ThinkingConfig(thinking_budget=0) if thinking_level == "none" else genai.types.ThinkingConfig(thinking_budget=100)
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=model_object_type,
            thinking_config=thinking_config
        )
    )
    return response.parsed

def generate_model_gpt(prompt: str, model_object_type: Optional[Type[BaseModel]] = None, system_prompt: str = "", image: bytes = None, multi_image: list[bytes] = None, image_detail: str = "low", reasoning_level: str = "minimal", model: str = "gpt-5-mini") -> BaseModel:
    return generate_model_gpt_with_cost(
        prompt, 
        model_object_type=model_object_type, 
        system_prompt=system_prompt, 
        image=image, 
        multi_image=multi_image, 
        image_detail=image_detail, 
        model=model, 
        reasoning_level=reasoning_level)[0]

def generate_model(prompt: str, model_object_type: Optional[Type[BaseModel]] = None, system_prompt: str = "", image: bytes = None, multi_image: list[bytes] = None, image_detail: str = "low", reasoning_level: str = "minimal", model: str = "gpt-5-mini") -> BaseModel:
    if model.startswith("gpt"):
        return generate_model_gpt(
            prompt=prompt, 
            model_object_type=model_object_type, 
            system_prompt=system_prompt, 
            image=image, 
            multi_image=multi_image, 
            image_detail=image_detail, reasoning_level=reasoning_level, model=model)
    elif model.startswith("gemini"):
        return generate_model_gemini(
            prompt=prompt, 
            model_object_type=model_object_type, 
            system_prompt=system_prompt, 
            image=image, 
            multi_image=multi_image, 
            thinking_level=reasoning_level, 
            model=model)
    else:
        raise ValueError(f"Invalid model: {model}")