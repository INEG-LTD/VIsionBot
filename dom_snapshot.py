from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

from models.core_models import DetectedElement, PageElements, PageInfo, PageSection

DOM_ELEMENT_CAPTURE_SCRIPT = """
() => {
    const selectors = [
        "button",
        "a[href]",
        "input:not([type='hidden'])",
        "textarea",
        "select",
        "[role='button']",
        "[role='link']",
        "[role='option']",
        "[role='menuitem']",
        "[role='tab']",
        "[role='checkbox']",
        "[role='radio']"
    ];

    const seen = new Set();
    const elements = [];
    const MAX_ELEMENTS = 800;

    const buildCssPath = (node) => {
        const parts = [];
        let current = node;
        while (current && current.nodeType === Node.ELEMENT_NODE) {
            let part = current.tagName.toLowerCase();
            if (current.id) {
                part += `#${current.id}`;
            } else if (current.className && typeof current.className === 'string') {
                const classes = current.className.trim().split(/\\s+/).slice(0, 2);
                if (classes.length) {
                    part += `.${classes.join('.')}`;
                }
            }
            parts.unshift(part);
            current = current.parentElement;
        }
        return parts.join(" ");
    };

    const addNode = (node) => {
        if (!node || seen.has(node)) {
            return;
        }
        seen.add(node);

        const rect = node.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) {
            return;
        }

        const textContent = (node.innerText || node.value || "").trim();
        const idx = elements.length + 1;
        node.dataset.domIndex = String(idx);
        elements.push({
            index: idx,
            tagName: node.tagName.toLowerCase(),
            textContent,
            text: textContent,
            ariaLabel: node.getAttribute("aria-label") || "",
            placeholder: node.getAttribute("placeholder") || "",
            title: node.getAttribute("title") || "",
            role: node.getAttribute("role") || "",
            type: node.getAttribute("type") || "",
            name: node.getAttribute("name") || "",
            href: node.href || "",
            className: node.className || "",
            boundingBox: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            },
            cssPath: buildCssPath(node),
        });
    };

    selectors.forEach((selector) => {
        const nodes = document.querySelectorAll(selector);
        nodes.forEach((node) => {
            addNode(node);
        });
    });
    return elements.slice(0, MAX_ELEMENTS);
}
"""


def _normalize_box(raw_box: Dict[str, float], page_info: PageInfo) -> List[int]:
    height = page_info.height or 1
    width = page_info.width or 1
    y_min = int(max(0, min(1000, (raw_box.get("y", 0) / height) * 1000)))
    x_min = int(max(0, min(1000, (raw_box.get("x", 0) / width) * 1000)))
    y_max = int(max(y_min, min(1000, (raw_box.get("y", 0) + raw_box.get("height", 0) / max(1, height)) * 1000)))
    x_max = int(max(x_min, min(1000, (raw_box.get("x", 0) + raw_box.get("width", 0) / max(1, width)) * 1000)))
    return [y_min, x_min, y_max, x_max]


def _describe_element(raw: Dict[str, Any]) -> str:
    parts: List[str] = []
    tag = raw.get("tagName") or "element"
    text = raw.get("textContent", "")
    role = raw.get("role", "")
    if text:
        parts.append(text)
    elif role:
        parts.append(role)
    else:
        parts.append(tag)
    if raw.get("placeholder"):
        parts.append(f'({raw["placeholder"]})')
    return " ".join(parts).strip()


def build_page_elements(raw_elements: List[Dict[str, Any]], page_info: PageInfo, max_elements: int = 400) -> PageElements:
    """Convert raw DOM capture into structured PageElements."""
    detected: List[DetectedElement] = []
    for raw in raw_elements[:max_elements]:
        bounding_box = raw.get("boundingBox", {})
        box = _normalize_box(bounding_box, page_info)
        element = DetectedElement(
            element_label=raw.get("ariaLabel") or raw.get("placeholder") or _describe_element(raw),
            description=_describe_element(raw),
            element_type=raw.get("tagName") or "element",
            is_clickable=True,
            box_2d=box,
            section=PageSection.CONTENT,
            field_subtype=raw.get("type") or raw.get("role") or None,
            confidence=0.65,
            requires_special_handling=False,
            overlay_number=raw.get("index"),
        )
        detected.append(element)
    return PageElements(elements=detected)


def capture_dom_elements(page, page_info: PageInfo, max_elements: int = 400) -> List[Dict[str, Any]]:
    """Capture interactive elements via DOM inspection and return serializable metadata."""
    raw = page.evaluate(DOM_ELEMENT_CAPTURE_SCRIPT)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    result = []
    for elem in raw[:max_elements]:
        norm = _normalize_box(elem.get("boundingBox", {}), page_info)
        elem["normalizedCoords"] = norm
        result.append(elem)
    return result
