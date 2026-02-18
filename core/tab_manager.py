"""
Tab Manager - Tracks browser tabs and dialogs.

Passively monitors Playwright BrowserContext events (new pages, closed pages, dialogs)
and provides methods for tab switching, closing, opening, and dialog dismissal.
Surfaces tab state for injection into the agent's prompt each iteration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, Page, Dialog


@dataclass
class TabInfo:
    """Metadata for a tracked browser tab."""
    id: str               # Stable ID like "t1", "t2" — never reused
    page: "Page"          # Playwright Page reference
    opener_id: Optional[str] = None  # If spawned by another tab
    created_at: float = field(default_factory=time.time)
    url: str = ""
    title: str = ""


@dataclass
class DialogInfo:
    """A JavaScript dialog being held open."""
    dialog: "Dialog"      # Playwright Dialog object (not yet accepted/dismissed)
    type: str             # "alert", "confirm", "prompt", "beforeunload"
    message: str
    default_value: str
    tab_id: str           # Which tab owns this dialog


class TabManager:
    """
    Tracks all open browser tabs and pending dialogs.

    Hooks into the Playwright BrowserContext's page lifecycle events.
    The agent interacts with tabs through switch_to/close_tab/open_tab,
    and the prompt builder reads state via build_tab_bar/build_dialog_notice.
    """

    def __init__(self, context: "BrowserContext", active_page: "Page") -> None:
        self._context = context
        self._next_id = 1
        self._tabs: List[TabInfo] = []
        self._active_id: Optional[str] = None
        self.pending_dialog: Optional[DialogInfo] = None
        self.tab_events: List[str] = []

        # Newly created pages are queued and finalized on the main thread.
        # This avoids cross-thread Playwright calls while still filtering
        # pages that close immediately (e.g. download popups).
        self._debounce_window_seconds = 0.5
        self._pending_pages: Dict[int, Tuple["Page", Optional[str], float]] = {}

        # Register the initial page
        initial = self._make_tab(active_page)
        self._tabs.append(initial)
        self._active_id = initial.id
        self._attach_page_listeners(active_page, initial.id)

        # Listen for new pages on the context
        context.on("page", self._on_page_created)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_tab(self, page: "Page", opener_id: Optional[str] = None) -> TabInfo:
        tab_id = f"t{self._next_id}"
        self._next_id += 1
        url = ""
        title = ""
        try:
            url = page.url
            title = page.title()
        except Exception:
            pass
        return TabInfo(
            id=tab_id,
            page=page,
            opener_id=opener_id,
            url=url,
            title=title,
        )

    def _attach_page_listeners(self, page: "Page", tab_id: str) -> None:
        page.on("close", lambda: self._on_page_closed(page))
        page.on("dialog", lambda dialog: self._on_dialog(dialog, tab_id))

    def _find_tab_by_page(self, page: "Page") -> Optional[TabInfo]:
        for tab in self._tabs:
            if tab.page is page:
                return tab
        return None

    def _find_tab_by_id(self, tab_id: str) -> Optional[TabInfo]:
        for tab in self._tabs:
            if tab.id == tab_id:
                return tab
        return None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_page_created(self, page: "Page") -> None:
        """Called when a new page (tab/popup) is created in the context."""
        # If already tracked (e.g. open_tab() registered it immediately), skip.
        if self._find_tab_by_page(page):
            return

        # Determine opener
        opener_id = None
        try:
            opener_page = page.opener
            if opener_page:
                opener_tab = self._find_tab_by_page(opener_page)
                if opener_tab:
                    opener_id = opener_tab.id
        except Exception:
            pass

        # Queue for debounce finalization on the main thread.
        self._pending_pages[id(page)] = (page, opener_id, time.time())

    def _finalize_pending_pages(self) -> None:
        """Register debounced pages from the main loop thread."""
        if not self._pending_pages:
            return

        now = time.time()
        matured_refs = [
            page_ref
            for page_ref, (_, _, created_at) in self._pending_pages.items()
            if now - created_at >= self._debounce_window_seconds
        ]
        if not matured_refs:
            return

        for page_ref in matured_refs:
            entry = self._pending_pages.pop(page_ref, None)
            if not entry:
                continue
            page, opener_id, _ = entry

            # open_tab() may have registered this already.
            if self._find_tab_by_page(page):
                continue

            try:
                if page.is_closed():
                    self.tab_events.append("A tab briefly opened and closed (possible download)")
                    continue
            except Exception:
                self.tab_events.append("A tab briefly opened and closed (possible download)")
                continue

            tab = self._make_tab(page, opener_id=opener_id)
            self._tabs.append(tab)
            self._active_id = tab.id

            try:
                self._attach_page_listeners(page, tab.id)
            except Exception:
                # Page may close between the is_closed() check and listener attach.
                self._tabs = [t for t in self._tabs if t.id != tab.id]
                self.tab_events.append("A tab briefly opened and closed (possible download)")
                continue

            opener_note = f" (opened by {opener_id})" if opener_id else ""
            title = tab.title or tab.url or "blank"
            self.tab_events.append(f"New tab opened: [{tab.id}] {title}{opener_note}")

    def _on_page_closed(self, page: "Page") -> None:
        """Called when a page closes."""
        tab = self._find_tab_by_page(page)
        if not tab:
            return

        closed_id = tab.id
        was_active = (self._active_id == closed_id)

        # Remove from registry
        self._tabs = [t for t in self._tabs if t.id != closed_id]
        self.tab_events.append(f"Tab [{closed_id}] closed")

        if was_active:
            # Apply fallback to pick new active tab
            new_active = self._pick_fallback(tab)
            if new_active:
                self._active_id = new_active.id
                self.tab_events.append(
                    f"Switched to tab [{new_active.id}] (tab [{closed_id}] was closed)"
                )
            else:
                self._active_id = None

    def _pick_fallback(self, closed_tab: TabInfo) -> Optional[TabInfo]:
        """Pick the next active tab after one closes."""
        if not self._tabs:
            return None

        # 1. Prefer opener if still open
        if closed_tab.opener_id:
            opener = self._find_tab_by_id(closed_tab.opener_id)
            if opener:
                return opener

        # 2. Pick adjacent tab (prefer left, then right)
        # Since the closed tab is already removed, just pick the last tab
        return self._tabs[-1] if self._tabs else None

    def _on_dialog(self, dialog: "Dialog", tab_id: str) -> None:
        """Called when a JavaScript dialog appears. Holds it open for the agent."""
        self.pending_dialog = DialogInfo(
            dialog=dialog,
            type=dialog.type,
            message=dialog.message,
            default_value=dialog.default_value or "",
            tab_id=tab_id,
        )
        self.tab_events.append(
            f"Dialog appeared on tab [{tab_id}]: \"{dialog.message}\" (type: {dialog.type})"
        )

    # ------------------------------------------------------------------
    # Public API — tab operations
    # ------------------------------------------------------------------

    def get_active(self) -> Optional[TabInfo]:
        if self._active_id is None:
            return None
        return self._find_tab_by_id(self._active_id)

    def get_all(self) -> List[TabInfo]:
        return list(self._tabs)

    def switch_to(self, tab_id: str) -> "Page":
        """Switch to a tab by ID. Returns the Page object."""
        tab = self._find_tab_by_id(tab_id)
        if not tab:
            raise ValueError(f"No tab with id '{tab_id}'. Open tabs: {[t.id for t in self._tabs]}")

        self._active_id = tab.id
        # Bring to front
        try:
            tab.page.bring_to_front()
        except Exception:
            pass

        # Wait for content to be ready (3s timeout)
        try:
            tab.page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass  # Screenshot anyway if timeout

        return tab.page

    def close_tab(self, tab_id: str) -> "Page":
        """Close a tab. Returns the page that becomes active."""
        tab = self._find_tab_by_id(tab_id)
        if not tab:
            raise ValueError(f"No tab with id '{tab_id}'")

        if len(self._tabs) <= 1:
            raise ValueError("Cannot close the last remaining tab")

        # Close triggers _on_page_closed which handles fallback
        tab.page.close()

        # Return the new active page
        active = self.get_active()
        if active:
            try:
                active.page.bring_to_front()
            except Exception:
                pass
            return active.page
        raise RuntimeError("No tabs remaining after close")

    def open_tab(self, url: Optional[str] = None) -> "Page":
        """Open a new tab, optionally navigating to a URL. Returns the new Page."""
        # Keep registry current before forcing a new tab.
        self._finalize_pending_pages()
        new_page = self._context.new_page()

        # The context "page" event may have queued this page for debounce.
        self._pending_pages.pop(id(new_page), None)

        # Check if already registered by the context page event handler.
        existing = self._find_tab_by_page(new_page)
        if not existing:
            tab = self._make_tab(new_page)
            self._tabs.append(tab)
            self._attach_page_listeners(new_page, tab.id)
            self._active_id = tab.id
            self.tab_events.append(f"Opened new tab [{tab.id}]")

        if url:
            try:
                new_page.goto(url, wait_until="domcontentloaded", timeout=10000)
            except Exception:
                pass  # Page may still be usable

        return new_page

    def dismiss_dialog(self, accept: bool, input_text: Optional[str] = None) -> None:
        """Accept or dismiss the pending dialog."""
        if not self.pending_dialog:
            return

        dialog = self.pending_dialog.dialog
        try:
            if accept:
                if input_text is not None:
                    dialog.accept(input_text)
                else:
                    dialog.accept(self.pending_dialog.default_value)
            else:
                dialog.dismiss()
        except Exception:
            pass

        action = "accepted" if accept else "dismissed"
        self.tab_events.append(f"Dialog {action}: \"{self.pending_dialog.message}\"")
        self.pending_dialog = None

    # ------------------------------------------------------------------
    # Public API — state for prompt injection
    # ------------------------------------------------------------------

    def refresh_metadata(self) -> None:
        """Update url/title for all tabs from their Page objects."""
        self._finalize_pending_pages()

        for tab in self._tabs:
            try:
                if not tab.page.is_closed():
                    tab.url = tab.page.url
                    tab.title = tab.page.title()
            except Exception:
                pass

    def build_tab_bar(self) -> Optional[str]:
        """Format the tab bar for prompt injection. Returns None if only 1 tab."""
        if len(self._tabs) <= 1:
            return None

        lines = []
        for tab in self._tabs:
            is_active = (tab.id == self._active_id)
            marker = " \u2192 " if is_active else "   "
            title = tab.title or "untitled"
            url = tab.url or ""

            # Truncate long URLs
            if len(url) > 60:
                url = url[:57] + "..."

            opener_note = f" (opened by {tab.opener_id})" if tab.opener_id else ""
            active_note = "    \u2190 active" if is_active else ""

            # Check for dialog on this tab
            dialog_note = ""
            if self.pending_dialog and self.pending_dialog.tab_id == tab.id and not is_active:
                dialog_note = " (dialog)"

            lines.append(f"{marker}[{tab.id}] {title} ({url}){opener_note}{dialog_note}{active_note}")

        result = "\n".join(lines)

        # Nudge if too many tabs
        if len(self._tabs) >= 5:
            result += f"\n\n{len(self._tabs)} tabs open \u2014 consider closing tabs you no longer need."

        return result

    def build_dialog_notice(self) -> Optional[str]:
        """Build dialog warning text if a dialog is pending on the active tab."""
        if not self.pending_dialog:
            return None
        if self.pending_dialog.tab_id != self._active_id:
            return None

        d = self.pending_dialog
        notice = f"\u26a0 DIALOG ({d.type}): \"{d.message}\"\n"

        if d.type == "alert":
            notice += "  Only action: dismiss_dialog(accept=true)\n"
        elif d.type == "confirm":
            notice += "  dismiss_dialog(accept=true) = OK\n"
            notice += "  dismiss_dialog(accept=false) = Cancel\n"
        elif d.type == "prompt":
            notice += f"  Default value: \"{d.default_value}\"\n"
            notice += "  dismiss_dialog(accept=true, input_text=\"...\") = submit\n"
            notice += "  dismiss_dialog(accept=false) = cancel\n"
        elif d.type == "beforeunload":
            notice += "  dismiss_dialog(accept=true) = Leave page\n"
            notice += "  dismiss_dialog(accept=false) = Stay on page\n"

        return notice

    def drain_events(self) -> List[str]:
        """Return and clear accumulated tab events."""
        events = self.tab_events.copy()
        self.tab_events.clear()
        return events

    def has_pending_dialog_on_active(self) -> bool:
        """Whether the active tab has a blocking dialog."""
        if not self.pending_dialog:
            return False
        return self.pending_dialog.tab_id == self._active_id
