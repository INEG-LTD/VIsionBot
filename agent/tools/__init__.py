from __future__ import annotations

from agent.tooling.registry import ToolRegistry

from . import browser, communication, cognitive, data, navigation, skills, tabs


def _all_tool_functions():
    return [
        # Browser
        browser.click,
        browser.type_text,
        browser.clear_text,
        browser.select_option,
        browser.upload_file,
        browser.press_key,
        # Navigation
        navigation.open_url,
        navigation.go_back,
        navigation.go_forward,
        navigation.scroll_down,
        navigation.scroll_up,
        navigation.scroll_container,
        navigation.scroll_to_element,
        # Data
        data.extract_data,
        data.report_data,
        data.write_data,
        data.read_file,
        data.find_files,
        data.read_clipboard,
        # Cognitive
        cognitive.think,
        cognitive.assert_condition,
        cognitive.flag,
        cognitive.wait_for,
        # Skills
        skills.activate_skill,
        # Tabs
        tabs.switch_tab,
        tabs.close_tab,
        tabs.open_tab,
        tabs.dismiss_dialog,
        # Communication
        communication.ask_user,
        communication.send_email,
        communication.bash,
    ]


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for fn in _all_tool_functions():
        registry.register(fn)
    return registry


default_registry = create_default_registry()


__all__ = ["create_default_registry", "default_registry"]
