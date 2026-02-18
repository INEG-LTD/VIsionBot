from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

from models.models import DetectedElement, PageElements, PageInfo, PageSection

DOM_ELEMENT_CAPTURE_SCRIPT = """
() => {
    const selectors = [
        "button",
        "a[href]",
        "input:not([type='hidden'])",
        // Add support for contenteditable <div> elements (rich text editors, editable fields)
        "div[contenteditable='true']",
        "div[contenteditable='']",
        "div[contenteditable]",  // handles case-insensitivity and existence of attribute
        "textarea",
        "select",
        "[role='button']",
        "[role='link']",
        "[role='option']",
        "[role='menuitem']",
        "[role='tab']",
        "[role='checkbox']",
        "[role='radio']",

        // Headings in links (covers Google search, news sites, blogs)
        "a[href] h1",
        "a[href] h2",
        "a[href] h3",
        "a[href] h4",
        "a[href] h5",
        "a[href] h6",

        // Button text (covers Material-UI, Bootstrap, Tailwind styled buttons)
        "button span",
        "[role='button'] span",

        // Icons (covers icon buttons, navigation)
        "button svg",
        "a[href] svg",
        "button i",
        "a[href] i",

        // Images (covers thumbnails, product images)
        "a[href] img"
    ];

    const seen = new Set();
    const elements = [];
    const MAX_ELEMENTS = 800;
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
    const vpArea = viewportWidth * viewportHeight;

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

        // Keep only elements that are actually visible in the current viewport.
        // This prevents the planner from targeting off-screen entries.
        const isInViewport = (
            rect.right > 0 &&
            rect.bottom > 0 &&
            rect.left < viewportWidth &&
            rect.top < viewportHeight
        );
        if (!isInViewport) {
            return;
        }

        const style = window.getComputedStyle(node);
        if (!style || style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
            return;
        }

        if (rect.width <= 0 || rect.height <= 0) {
            return;
        }

        const textContent = (node.innerText || node.value || "").trim();
        const idx = elements.length + 1;
        node.dataset.domIndex = String(idx);
        // Check if this element or any of its descendants is currently focused
        const activeElement = document.activeElement;
        const isFocused = activeElement === node || (activeElement && node.contains && node.contains(activeElement));
        elements.push({
            index: idx,
            tagName: node.tagName.toLowerCase(),
            textContent,
            text: textContent,
            ariaLabel: node.getAttribute("aria-label") || "",
            alt: node.getAttribute("alt") || "",
            placeholder: node.getAttribute("placeholder") || "",
            title: node.getAttribute("title") || "",
            role: node.getAttribute("role") || "",
            type: node.getAttribute("type") || "",
            name: node.getAttribute("name") || "",
            href: node.href || "",
            className: node.className || "",
            id: node.id || "",
            boundingBox: {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height
            },
            cssPath: buildCssPath(node),
            isFocused: isFocused || false,
            bvbDone: node.dataset.bvbDone === "true",
        });
    };

    // ── Pass 1: Selector-based detection (semantic HTML) ──
    selectors.forEach((selector) => {
        const nodes = document.querySelectorAll(selector);
        nodes.forEach((node) => {
            addNode(node);
        });
    });

    // ── Pass 2: interactivity-signal detection ──
    // Catches non-semantic interactive elements (custom components, styled divs,
    // draggable pieces, etc.) that Pass 1's CSS selectors can't reach.

    // Interactive cursor values (not just 'pointer')
    const interactiveCursors = new Set([
        'pointer', 'grab', 'grabbing', 'move', 'cell', 'copy', 'alias'
    ]);

    // Semantic interactive tags — their children are always subordinate
    // (clicking a span inside a button = clicking the button).
    // Pass 2 containers (divs etc.) are different — children may be independent.
    const semanticTags = new Set([
        'BUTTON', 'A', 'INPUT', 'TEXTAREA', 'SELECT', 'OPTION',
        'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
    ]);
    const interactiveRoles = new Set([
        'button', 'link', 'option', 'menuitem', 'tab', 'checkbox', 'radio',
    ]);
    const isSemanticInteractive = (el) => {
        if (semanticTags.has(el.tagName)) return true;
        const role = el.getAttribute('role') || '';
        return interactiveRoles.has(role);
    };

    // Build ancestor set for fast "is this an ancestor of a captured element?" lookups
    const ancestorsOfCaptured = new Set();
    for (const capturedNode of seen) {
        let parent = capturedNode.parentElement;
        while (parent) {
            if (ancestorsOfCaptured.has(parent)) break;
            ancestorsOfCaptured.add(parent);
            parent = parent.parentElement;
        }
    }

    const allElements = document.body.getElementsByTagName('*');
    for (let i = 0; i < allElements.length; i++) {
        if (elements.length >= MAX_ELEMENTS) break;

        const node = allElements[i];
        if (seen.has(node)) continue;

        // Skip non-element tags that can't be interactive
        const tag = node.tagName;
        if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT' ||
            tag === 'META' || tag === 'LINK' || tag === 'BR' || tag === 'HR') continue;

        const style = window.getComputedStyle(node);
        if (!style) continue;

        // Check interactivity signals
        const hasCursorInteractive = interactiveCursors.has(style.cursor);
        const tabindexAttr = node.getAttribute('tabindex');
        const hasTabindex = tabindexAttr !== null && parseInt(tabindexAttr) >= 0;
        const hasAriaInteraction = node.hasAttribute('aria-expanded') ||
                                   node.hasAttribute('aria-pressed') ||
                                   node.hasAttribute('aria-checked') ||
                                   node.hasAttribute('aria-haspopup');

        if (!hasCursorInteractive && !hasTabindex && !hasAriaInteraction) continue;

        // Skip if pointer-events: none (clicks pass through)
        if (style.pointerEvents === 'none') continue;

        // Visibility checks
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;

        const rect = node.getBoundingClientRect();
        if (rect.width <= 2 || rect.height <= 2) continue;

        // Viewport check
        if (rect.right <= 0 || rect.bottom <= 0 ||
            rect.left >= viewportWidth || rect.top >= viewportHeight) continue;

        // Skip if too large (>25% of viewport — likely a container, not a button)
        if (rect.width * rect.height > vpArea * 0.25) continue;

        // Skip if this is an ANCESTOR of a captured element
        // (children from Pass 1 are better targets)
        if (ancestorsOfCaptured.has(node)) continue;

        // Skip if this is a DESCENDANT of a SEMANTIC interactive element
        // (button, a, input, [role=button], etc. — children are subordinate).
        // But allow descendants of Pass 2 containers (divs) — their children
        // may be independently interactive (e.g. bot cards inside an accordion).
        let skipAsDescendant = false;
        let anc = node.parentElement;
        while (anc) {
            if (seen.has(anc) && isSemanticInteractive(anc)) {
                skipAsDescendant = true;
                break;
            }
            anc = anc.parentElement;
        }
        if (skipAsDescendant) continue;

        // Text context lifting — but ONLY for invisible overlays.
        // An invisible overlay covers the same area as its parent (e.g. chess.com's
        // toggleClickArea). A small visual element (chess piece, icon) should be
        // captured as-is even if it has no text.
        let target = node;
        const nodeText = (node.innerText || node.value || '').trim();
        if (!nodeText && node.parentElement) {
            const parentRect = node.parentElement.getBoundingClientRect();
            const isOverlay = (
                Math.abs(rect.width - parentRect.width) < 20 &&
                Math.abs(rect.height - parentRect.height) < 20
            );
            if (isOverlay) {
                let parent = node.parentElement;
                while (parent && parent !== document.body) {
                    if (seen.has(parent) || ancestorsOfCaptured.has(parent)) break;
                    const parentText = (parent.innerText || '').trim();
                    if (parentText) {
                        const pRect = parent.getBoundingClientRect();
                        if (pRect.width * pRect.height < vpArea * 0.25) {
                            target = parent;
                            break;
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }

        // If we lifted to a different node, verify it's not already captured
        if (target !== node && seen.has(target)) continue;

        addNode(target);
    }

    // ── Pass 3: Dynamically appeared elements (MutationObserver) ──
    // Catches elements that appeared after user actions — e.g. hint dots
    // on a chess board after clicking a piece, dropdown options after clicking
    // a trigger, etc. These may have no cursor:pointer or ARIA attributes,
    // but their appearance in response to an action implies interactivity.
    //
    // Two sets persist across captures:
    //   __bvb_newNodes  — freshly mutated nodes (filled by observer, drained each capture)
    //   __bvb_known     — nodes previously captured by Pass 3 (re-checked each capture)
    // This ensures dynamically added elements stay detected across iterations
    // as long as they remain in the DOM and visible.

    // Set up MutationObserver once (persists across captures)
    if (!window.__bvb_observer) {
        window.__bvb_newNodes = new Set();
        window.__bvb_known = new Set();
        window.__bvb_observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const added of mutation.addedNodes) {
                    if (added.nodeType !== Node.ELEMENT_NODE) continue;
                    window.__bvb_newNodes.add(added);
                    const desc = added.getElementsByTagName('*');
                    for (let j = 0; j < desc.length; j++) {
                        window.__bvb_newNodes.add(desc[j]);
                    }
                }
            }
        });
        window.__bvb_observer.observe(document.body, {
            childList: true,
            subtree: true,
        });
    }

    // Merge new mutations into the known set, then drain newNodes
    const newNodes = window.__bvb_newNodes;
    if (newNodes && newNodes.size > 0) {
        for (const n of newNodes) {
            window.__bvb_known.add(n);
        }
        window.__bvb_newNodes = new Set();
    }

    const knownPass3 = window.__bvb_known;
    if (knownPass3 && knownPass3.size > 0) {
        // Rebuild ancestor set to include Pass 2 additions
        const ancestorsOfAll = new Set();
        for (const capturedNode of seen) {
            let parent = capturedNode.parentElement;
            while (parent) {
                if (ancestorsOfAll.has(parent)) break;
                ancestorsOfAll.add(parent);
                parent = parent.parentElement;
            }
        }

        // Prune nodes no longer in DOM
        for (const node of knownPass3) {
            if (!document.body.contains(node)) {
                knownPass3.delete(node);
            }
        }

        for (const node of knownPass3) {
            if (elements.length >= MAX_ELEMENTS) break;
            if (seen.has(node)) continue;

            const tag = node.tagName;
            if (!tag) continue;
            if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT' ||
                tag === 'META' || tag === 'LINK' || tag === 'BR' || tag === 'HR') continue;

            const style = window.getComputedStyle(node);
            if (!style) continue;
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
            // NOTE: Do NOT filter pointer-events:none here. Pass 3 elements
            // appeared after an action — their presence IS the interactivity
            // signal (e.g. chess hint dots). The agent clicks at the location
            // and the click passes through to the interactive element below.

            const rect = node.getBoundingClientRect();
            if (rect.width <= 2 || rect.height <= 2) continue;

            if (rect.right <= 0 || rect.bottom <= 0 ||
                rect.left >= viewportWidth || rect.top >= viewportHeight) continue;

            if (rect.width * rect.height > vpArea * 0.25) continue;

            // Skip if ancestor of a captured element
            if (ancestorsOfAll.has(node)) continue;

            // Skip if descendant of a semantic interactive element
            let skipAsDescendant = false;
            let anc = node.parentElement;
            while (anc) {
                if (seen.has(anc) && isSemanticInteractive(anc)) {
                    skipAsDescendant = true;
                    break;
                }
                anc = anc.parentElement;
            }
            if (skipAsDescendant) continue;

            addNode(node);
        }
    }

    return elements.slice(0, MAX_ELEMENTS);
}
"""


def _normalize_box(raw_box: Dict[str, float], page_info: PageInfo) -> List[int]:
    height = page_info.height or 1
    width = page_info.width or 1
    y_min = int(max(0, min(1000, (raw_box.get("y", 0) / height) * 1000)))
    x_min = int(max(0, min(1000, (raw_box.get("x", 0) / width) * 1000)))
    y_max = int(max(y_min, min(1000, ((raw_box.get("y", 0) + raw_box.get("height", 0)) / height) * 1000)))                          
    x_max = int(max(x_min, min(1000, ((raw_box.get("x", 0) + raw_box.get("width", 0)) / width) * 1000)))
    return [y_min, x_min, y_max, x_max]


def _describe_element(raw: Dict[str, Any]) -> str:
    parts: List[str] = []
    tag = raw.get("tagName") or "element"
    text = raw.get("textContent", "")
    alt = raw.get("alt", "")
    role = raw.get("role", "")
    if text:
        parts.append(text)
    elif alt:
        parts.append(alt)
    elif role:
        parts.append(role)
    else:
        parts.append(tag)
    if raw.get("placeholder"):
        parts.append(f'({raw["placeholder"]})')
    return " ".join(parts).strip()


def _clean_class(raw_class) -> Optional[str]:
    """Extract first few CSS classes, skip empty/dict values."""
    if not raw_class or isinstance(raw_class, dict):
        return None
    cls = str(raw_class).strip()
    if not cls:
        return None
    # Keep first 3 classes to avoid huge strings
    parts = cls.split()[:3]
    return " ".join(parts) if parts else None


def _text_presence_score(raw: Dict[str, Any]) -> int:
    """
    Score how much visible text signal this element provides.
    3: inner text/value present
    2: placeholder/title/alt present
    1: aria-label only
    0: no text signal
    """
    text = (raw.get("textContent") or "").strip()
    placeholder = (raw.get("placeholder") or "").strip()
    title = (raw.get("title") or "").strip()
    alt = (raw.get("alt") or "").strip()
    aria = (raw.get("ariaLabel") or "").strip()

    if text:
        return 3
    if placeholder or title or alt:
        return 2
    if aria:
        return 1
    return 0


def build_page_elements(page, page_info: PageInfo) -> PageElements:
    """Convert raw DOM capture into structured PageElements."""
    
    def is_clickable(raw: Dict[str, Any]) -> bool:
        # All elements captured by the DOM script are interactive —
        # Pass 1 matches semantic elements, Pass 2 matches cursor:pointer / ARIA.
        return True

    detected: List[DetectedElement] = []
    
    raw_elements = page.evaluate(DOM_ELEMENT_CAPTURE_SCRIPT)
    if isinstance(raw_elements, str):
        try:
            raw_elements = json.loads(raw_elements)
        except Exception:
            return PageElements(elements=[])
    if not isinstance(raw_elements, list):
        return PageElements(elements=[])  
    for raw in raw_elements:
        bounding_box = raw.get("boundingBox", {})
        box = _normalize_box(bounding_box, page_info)
        text_score = _text_presence_score(raw)
        element = DetectedElement(
            element_label=raw.get("ariaLabel") or raw.get("alt") or raw.get("placeholder") or _describe_element(raw),
            description=_describe_element(raw),
            element_type=raw.get("tagName") or "element",
            is_clickable=is_clickable(raw),
            box_2d=box,
            section=PageSection.CONTENT,
            field_subtype=raw.get("type") or raw.get("role") or None,
            confidence=0.65,
            requires_special_handling=False,
            overlay_number=raw.get("index"),
            is_focused=raw.get("isFocused", False),
            has_visible_text=text_score >= 2,
            text_presence_score=text_score,
            css_class=_clean_class(raw.get("className")),
            css_id=raw.get("id") or None,
            is_done=raw.get("bvbDone", False),
        )
        detected.append(element)
    return PageElements(elements=detected)


def capture_dom_elements(page, page_info: PageInfo, max_elements: int = 400) -> PageElements:
    """Capture interactive elements via DOM inspection and return serializable metadata."""
    raw = page.evaluate(DOM_ELEMENT_CAPTURE_SCRIPT)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return PageElements(elements=[])
    if not isinstance(raw, list):
        return PageElements(elements=[])   
    elements = []
    for elem in raw[:max_elements]:
        bounding_box = elem.get("boundingBox", {})
        box = _normalize_box(bounding_box, page_info)
        element = DetectedElement(
            element_label=elem.get("ariaLabel") or elem.get("placeholder") or _describe_element(elem),
            description=_describe_element(elem),
            element_type=elem.get("tagName") or "element",
            is_clickable=True,
            box_2d=box,
            section=PageSection.CONTENT,
            is_focused=elem.get("isFocused", False),
        )
        elements.append(element)
    return PageElements(elements=elements)
