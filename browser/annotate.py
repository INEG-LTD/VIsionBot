"""
Element Index + Crop Gallery for element_index interaction mode.

Provides:
- build_element_index(): text index of interactive elements grouped by region
- build_crop_gallery(): visual gallery of text-poor elements
- describe_position(): region name from box_2d coordinates
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

import math

from models.models import DetectedElement, PageElements, PageInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_to_pixels(
    box_2d: List[int],
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """Convert normalised [y_min, x_min, y_max, x_max] (0-1000) → pixel coords."""
    y_min, x_min, y_max, x_max = box_2d
    px_x0 = int(x_min / 1000 * img_w)
    px_y0 = int(y_min / 1000 * img_h)
    px_x1 = int(x_max / 1000 * img_w)
    px_y1 = int(y_max / 1000 * img_h)
    return px_x0, px_y0, px_x1, px_y1


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a bold monospace font; fall back to default."""
    import platform

    candidates: list[str] = []
    if platform.system() == "Darwin":
        candidates = [
            "/System/Library/Fonts/SFCompact.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    # Ultimate fallback
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Element Index helpers
# ---------------------------------------------------------------------------

def describe_position(box_2d: List[int]) -> str:
    """Convert box_2d [y_min, x_min, y_max, x_max] (0-1000) to a region name."""
    y_min, x_min, y_max, x_max = box_2d
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    h = "left" if cx < 333 else ("right" if cx > 666 else "center")
    v = "top" if cy < 333 else ("bottom" if cy > 666 else "middle")

    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return h
    if h == "center":
        return f"{v}-center"
    return f"{v}-{h}"


# ---------------------------------------------------------------------------
# Element Index builder
# ---------------------------------------------------------------------------

@dataclass
class ElementIndexResult:
    """Result of build_element_index()."""
    index_text: str
    text_poor_elements: List[DetectedElement] = field(default_factory=list)


def build_element_index(
    elements: List[DetectedElement],
    max_elements: int = 0,
    viewport_only: bool = False,
) -> ElementIndexResult:
    """Build a text index of interactive elements grouped by screen region.

    Text-rich elements (text_presence_score >= 2) get description text.
    Text-poor elements (score <= 1) get a 'SEE CROP GALLERY' tag.
    Focused elements are extracted into a [FOCUSED] section at the top.

    Args:
        elements: Detected page elements.
        max_elements: Hard cap on elements included (0 = unlimited).
        viewport_only: If True, skip elements entirely outside 0-1000 viewport.

    Returns:
        ElementIndexResult with the index text and the list of text-poor elements.
    """
    # Filter to elements with overlay numbers
    candidates = [e for e in elements if getattr(e, 'overlay_number', None) is not None]

    # Viewport filter
    if viewport_only:
        def _in_viewport(el: DetectedElement) -> bool:
            box = getattr(el, 'box_2d', None)
            if not box or len(box) != 4:
                return False
            y_min, x_min, y_max, x_max = box
            return x_max > 0 and y_max > 0 and x_min < 1000 and y_min < 1000
        candidates = [e for e in candidates if _in_viewport(e)]

    # Hard cap: sort by prominence (area * (text_score + 1)), keep top N
    overflow_count = 0
    if max_elements > 0 and len(candidates) > max_elements:
        def _prominence(el: DetectedElement) -> float:
            box = getattr(el, 'box_2d', None)
            if not box or len(box) != 4:
                return 0
            y_min, x_min, y_max, x_max = box
            area = max(0, x_max - x_min) * max(0, y_max - y_min)
            score = int(getattr(el, 'text_presence_score', 0) or 0)
            return area * (score + 1)
        candidates.sort(key=_prominence, reverse=True)
        overflow_count = len(candidates) - max_elements
        candidates = candidates[:max_elements]

    # Split into text-rich / text-poor
    text_rich: List[DetectedElement] = []
    text_poor: List[DetectedElement] = []
    for elem in candidates:
        score = int(getattr(elem, 'text_presence_score', 0) or 0)
        if score >= 2:
            text_rich.append(elem)
        else:
            text_poor.append(elem)

    # Build per-element line
    def _build_line(elem: DetectedElement, gallery_tag: bool) -> str:
        etype = elem.element_type or "element"
        subtype = f" [{elem.field_subtype}]" if elem.field_subtype else ""
        focused = " (focused)" if elem.is_focused else ""

        if gallery_tag:
            # Text-poor: show CSS identifier + gallery tag
            css_id = getattr(elem, 'css_id', None) or ""
            css_class = getattr(elem, 'css_class', None) or ""
            ident = ""
            if css_id:
                ident = f" #{css_id}"
            elif css_class:
                ident = " ." + css_class.split()[0]
            label_text = elem.element_label or ""
            if label_text:
                ident += f' "{label_text}"'
            return f"  [{elem.overlay_number}] {etype}{ident}{focused} — SEE CROP GALLERY"
        else:
            # Text-rich: show label + CSS hint for disambiguation
            label = elem.element_label or ""
            css_hint = ""
            css_id = getattr(elem, 'css_id', None) or ""
            css_class = getattr(elem, 'css_class', None) or ""
            if css_id:
                css_hint = f" #{css_id}"
            elif css_class:
                css_hint = " ." + css_class.split()[0]
            return f"  [{elem.overlay_number}] {etype}{subtype}: {label}{css_hint}{focused}"

    # Group elements by region, separate focused ones
    focused_lines: List[str] = []
    region_groups: dict[str, List[str]] = {}

    for elem in text_rich:
        line = _build_line(elem, gallery_tag=False)
        if elem.is_focused:
            focused_lines.append(line)
        region = describe_position(elem.box_2d)
        region_groups.setdefault(region, []).append(line)

    for elem in text_poor:
        line = _build_line(elem, gallery_tag=True)
        if elem.is_focused:
            focused_lines.append(line)
        region = describe_position(elem.box_2d)
        region_groups.setdefault(region, []).append(line)

    # Render index text
    index_lines: List[str] = []

    # Focused section at top
    if focused_lines:
        index_lines.append("[FOCUSED]")
        index_lines.extend(focused_lines)

    # Regional sections
    region_order = [
        "top-left", "top-center", "top-right",
        "left", "center", "right",
        "bottom-left", "bottom-center", "bottom-right",
    ]
    for region in region_order:
        if region in region_groups:
            index_lines.append(f"[{region.upper()}]")
            index_lines.extend(region_groups[region])
    # Any regions not in the predefined order
    for region, lines in region_groups.items():
        if region not in region_order:
            index_lines.append(f"[{region.upper()}]")
            index_lines.extend(lines)

    if overflow_count > 0:
        index_lines.append(f"...and {overflow_count} more elements (not shown)")

    index_text = "\n".join(index_lines) if index_lines else "No interactive elements detected."

    return ElementIndexResult(
        index_text=index_text,
        text_poor_elements=text_poor,
    )


# ---------------------------------------------------------------------------
# Crop Gallery for Element Index mode
# ---------------------------------------------------------------------------

_GALLERY_BG = (30, 30, 30)            # dark background
_GALLERY_CELL_BG = (50, 50, 50)       # slightly lighter cell background
_GALLERY_ID_BG = (0, 120, 220)        # blue pill for [id] label
_GALLERY_POS_COLOR = (180, 180, 180)  # gray for position text
_GALLERY_ID_COLOR = (255, 255, 255)   # white for [id] text
_GALLERY_TITLE_COLOR = (200, 200, 200)


def build_crop_gallery(
    screenshot_bytes: bytes,
    elements: List[DetectedElement],
    crops_per_page: int = 6,
    crop_cell_size: int = 200,
    cell_padding: int = 24,
) -> List[bytes]:
    """Build gallery page images showing cropped text-poor elements.

    Each gallery page is a grid of cropped elements with large [id] labels
    and position descriptions.  Pages use 2 columns with generous spacing.
    There is no limit on the number of pages produced.

    Args:
        screenshot_bytes: Clean screenshot PNG bytes.
        elements: Text-poor elements to include (pre-filtered by caller).
        crops_per_page: How many crops per gallery image.
        crop_cell_size: Pixel size for the crop display area (width & height).
        cell_padding: Padding around each cell.

    Returns:
        List of PNG bytes, one per gallery page.  Empty list if no elements.
    """
    if not elements:
        return []

    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    img_w, img_h = img.size

    # Fonts
    id_font = _get_font(22)
    pos_font = _get_font(14)
    title_font = _get_font(16)

    # Layout constants
    cols = 2
    rows_per_page = math.ceil(crops_per_page / cols)
    id_label_h = 32       # height reserved for [id] label above crop
    pos_label_h = 22      # height reserved for position text below crop
    page_margin = 20
    title_h = 36           # title bar at top of each page

    full_cell_w = crop_cell_size + cell_padding * 2
    full_cell_h = id_label_h + crop_cell_size + pos_label_h + cell_padding * 2

    page_w = page_margin * 2 + cols * full_cell_w
    page_h = page_margin + title_h + rows_per_page * full_cell_h + page_margin

    # Split elements into pages
    pages_data: list[list[DetectedElement]] = []
    for i in range(0, len(elements), crops_per_page):
        pages_data.append(elements[i : i + crops_per_page])

    total_pages = len(pages_data)
    result_pages: list[bytes] = []

    for page_idx, page_elems in enumerate(pages_data):
        page_img = Image.new("RGB", (page_w, page_h), _GALLERY_BG)
        draw = ImageDraw.Draw(page_img)

        # Title
        title_text = f"CROP GALLERY — page {page_idx + 1}/{total_pages}"
        draw.text(
            (page_margin, page_margin + 6),
            title_text,
            fill=_GALLERY_TITLE_COLOR,
            font=title_font,
        )

        for cell_idx, elem in enumerate(page_elems):
            col = cell_idx % cols
            row = cell_idx // cols

            # Cell origin (top-left of the full cell area)
            cx0 = page_margin + col * full_cell_w
            cy0 = page_margin + title_h + row * full_cell_h

            # Cell background
            draw.rounded_rectangle(
                [
                    cx0 + cell_padding // 2,
                    cy0 + cell_padding // 2,
                    cx0 + full_cell_w - cell_padding // 2,
                    cy0 + full_cell_h - cell_padding // 2,
                ],
                radius=8,
                fill=_GALLERY_CELL_BG,
            )

            # --- [id] label ---
            id_text = f"[{elem.overlay_number}]"
            id_bbox = id_font.getbbox(id_text)
            id_tw = id_bbox[2] - id_bbox[0]
            id_th = id_bbox[3] - id_bbox[1]
            id_pill_w = id_tw + 16
            id_pill_h = id_th + 8
            id_x = cx0 + (full_cell_w - id_pill_w) // 2
            id_y = cy0 + cell_padding

            draw.rounded_rectangle(
                [id_x, id_y, id_x + id_pill_w, id_y + id_pill_h],
                radius=6,
                fill=_GALLERY_ID_BG,
            )
            draw.text(
                (id_x + 8, id_y + 4),
                id_text,
                fill=_GALLERY_ID_COLOR,
                font=id_font,
            )

            # --- Crop the element from the screenshot ---
            px_x0, px_y0, px_x1, px_y1 = _box_to_pixels(
                elem.box_2d, img_w, img_h
            )
            elem_w = max(1, px_x1 - px_x0)
            elem_h = max(1, px_y1 - px_y0)

            # Ensure minimum source crop area (tiny elements need context)
            MIN_CROP_SOURCE = 60
            if elem_w < MIN_CROP_SOURCE:
                expand_x = (MIN_CROP_SOURCE - elem_w) // 2
                px_x0 = max(0, px_x0 - expand_x)
                px_x1 = min(img_w, px_x1 + expand_x)
                elem_w = px_x1 - px_x0
            if elem_h < MIN_CROP_SOURCE:
                expand_y = (MIN_CROP_SOURCE - elem_h) // 2
                px_y0 = max(0, px_y0 - expand_y)
                px_y1 = min(img_h, px_y1 + expand_y)
                elem_h = px_y1 - px_y0

            # Add 25% context padding (min 30px)
            pad_x = max(30, int(elem_w * 0.25))
            pad_y = max(30, int(elem_h * 0.25))
            crop_x0 = max(0, px_x0 - pad_x)
            crop_y0 = max(0, px_y0 - pad_y)
            crop_x1 = min(img_w, px_x1 + pad_x)
            crop_y1 = min(img_h, px_y1 + pad_y)

            crop = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))

            # Scale to fit within crop_cell_size while keeping aspect ratio
            cw, ch = crop.size
            scale = min(crop_cell_size / max(1, cw), crop_cell_size / max(1, ch))
            new_w = max(1, int(cw * scale))
            new_h = max(1, int(ch * scale))
            crop = crop.resize((new_w, new_h), Image.LANCZOS)

            # Paste centred in the crop area
            crop_area_x = cx0 + cell_padding
            crop_area_y = cy0 + cell_padding + id_label_h
            paste_x = crop_area_x + (crop_cell_size - new_w) // 2
            paste_y = crop_area_y + (crop_cell_size - new_h) // 2
            page_img.paste(crop, (paste_x, paste_y))

            # --- Position label ---
            pos_text = describe_position(elem.box_2d)
            pos_bbox = pos_font.getbbox(pos_text)
            pos_tw = pos_bbox[2] - pos_bbox[0]
            pos_x = cx0 + (full_cell_w - pos_tw) // 2
            pos_y = crop_area_y + crop_cell_size + 4
            draw.text(
                (pos_x, pos_y), pos_text, fill=_GALLERY_POS_COLOR, font=pos_font
            )

        buf = io.BytesIO()
        page_img.save(buf, "PNG")
        result_pages.append(buf.getvalue())

    return result_pages
