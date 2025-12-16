import math
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from handlers.select_handler import SelectHandler
from models import ActionStep, PageElements, PageInfo, DetectedElement


FIXTURE_PATH = Path(__file__).parent / "select_fixtures.html"


def _normalized_box(box, doc_width, doc_height):
    """Convert Playwright bounding box to normalized Gemini-style box using document size."""
    x_min = box["x"]
    y_min = box["y"]
    x_max = x_min + box["width"]
    y_max = y_min + box["height"]
    return [
        int(y_min / doc_height * 1000),
        int(x_min / doc_width * 1000),
        int(y_max / doc_height * 1000),
        int(x_max / doc_width * 1000),
    ]


def _page_info(page):
    viewport = page.viewport_size or {"width": 1280, "height": 720}
    doc = page.evaluate(
        """() => ({
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            scrollX: window.scrollX,
            scrollY: window.scrollY,
            dpr: window.devicePixelRatio || 1,
            docWidth: document.documentElement.scrollWidth,
            docHeight: document.documentElement.scrollHeight,
        })"""
    )
    return PageInfo(
        width=viewport["width"],
        height=viewport["height"],
        scroll_x=doc["scrollX"],
        scroll_y=doc["scrollY"],
        url=page.url,
        title=page.title(),
        dpr=doc["dpr"],
        ss_pixel_w=viewport["width"],
        ss_pixel_h=viewport["height"],
        css_scale=1.0,
        doc_width=doc["docWidth"],
        doc_height=doc["docHeight"],
    )


def _detected_element(page, selector, overlay_number, description):
    box = page.locator(selector).bounding_box()
    assert box, f"Bounding box not found for selector {selector}"
    doc = page.evaluate(
        """() => ({
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            docWidth: document.documentElement.scrollWidth,
            docHeight: document.documentElement.scrollHeight,
        })"""
    )
    box_2d = _normalized_box(box, doc["docWidth"], doc["docHeight"])
    return DetectedElement(
        element_label=selector,
        description=description,
        element_type="select",
        is_clickable=True,
        box_2d=box_2d,
        section="content",
        field_subtype="select",
        confidence=0.9,
        requires_special_handling=True,
        overlay_number=overlay_number,
    )


def _run(handler, step, elements, page):
    # Use pixel center from the target element to guide coordinate-based resolution
    target = elements.elements[0]
    # Reconstruct the selector from the overlay_number mapping used to build the element
    # (in this test we know there is exactly one element and we can derive its center
    # directly from the DOM using its overlay-defined selector argument to _detected_element)
    # The description is not a selector, so we re-query the DOM via overlay index order.
    # For simplicity in this fixture, we map overlay_number to the target selector
    # by repeating the lookup that created the DetectedElement: the first element in
    # `elements` corresponds to the selector we passed to _detected_element.
    # We store that selector on the element for test purposes.
    selector = target.element_label
    if selector:
        bbox = page.locator(selector).bounding_box()
        if bbox:
            step.x = int(bbox["x"] + bbox["width"] / 2)
            step.y = int(bbox["y"] + bbox["height"] / 2)
    info = _page_info(page)
    handler.handle_select_field(step, elements, info)


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(FIXTURE_PATH.as_uri())
        yield page
        browser.close()


def test_basic_select(page):
    handler = SelectHandler(page)
    elem = _detected_element(page, "#basic-select", 1, "basic select")
    step = ActionStep(action="handle_select", overlay_index=1, select_option_text="Cherry")
    _run(handler, step, PageElements(elements=[elem]), page)
    value = page.eval_on_selector("#basic-select", "el => el.value")
    assert value == "cherry"


def test_placeholder_select(page):
    handler = SelectHandler(page)
    elem = _detected_element(page, "#placeholder-select", 2, "placeholder select")
    step = ActionStep(action="handle_select", overlay_index=2, select_option_text="Juice")
    _run(handler, step, PageElements(elements=[elem]), page)
    value = page.eval_on_selector("#placeholder-select", "el => el.value")
    assert value == "juice"


def test_optgroup_select(page):
    handler = SelectHandler(page)
    elem = _detected_element(page, "#optgroup-select", 3, "optgroup select")
    step = ActionStep(action="handle_select", overlay_index=3, select_option_text="Sandwich")
    _run(handler, step, PageElements(elements=[elem]), page)
    value = page.eval_on_selector("#optgroup-select", "el => el.value")
    assert value == "sandwich"


def test_multi_select(page):
    handler = SelectHandler(page)
    elem = _detected_element(page, "#multi-select", 4, "multi select")
    step = ActionStep(action="handle_select", overlay_index=4, select_option_text="Olives")
    _run(handler, step, PageElements(elements=[elem]), page)
    selected = page.eval_on_selector_all(
        "#multi-select option:checked", "opts => opts.map(o => o.value)"
    )
    assert "olives" in selected


def test_long_label_select(page):
    handler = SelectHandler(page)
    elem = _detected_element(page, "#long-label-select", 5, "long label select")
    step = ActionStep(action="handle_select", overlay_index=5, select_option_text="Production")
    _run(handler, step, PageElements(elements=[elem]), page)
    value = page.eval_on_selector("#long-label-select", "el => el.value")
    assert value == "prod"


def test_custom_dropdown_button_list(page):
    handler = SelectHandler(page)
    elem = _detected_element(page, "#custom-dropdown-trigger", 6, "custom dropdown button")
    step = ActionStep(action="handle_select", overlay_index=6, select_option_text="Green")
    _run(handler, step, PageElements(elements=[elem]), page)
    text = page.text_content("#custom-dropdown-trigger").strip()
    data_value = page.eval_on_selector("#custom-dropdown-trigger", "el => el.dataset.value || ''")
    assert text == "Green"
    assert data_value == "green"


def test_custom_listbox(page):
    handler = SelectHandler(page)
    elem = _detected_element(page, ".listbox-display", 7, "custom listbox div")
    step = ActionStep(action="handle_select", overlay_index=7, select_option_text="High")
    _run(handler, step, PageElements(elements=[elem]), page)
    text = page.text_content(".listbox-display").strip()
    data_value = page.eval_on_selector(".listbox-display", "el => el.dataset.value || ''")
    assert text == "High"
    assert data_value == "high"


def test_combobox_searchable(page):
    handler = SelectHandler(page)
    elem = _detected_element(page, "#combo-input", 8, "combobox input")
    step = ActionStep(action="handle_select", overlay_index=8, select_option_text="Dog")
    _run(handler, step, PageElements(elements=[elem]), page)
    value = page.eval_on_selector("#combo-input", "el => ({value: el.value, dataValue: el.dataset.value || ''})")
    assert value["value"] == "Dog"
    assert value["dataValue"] == "dog"




