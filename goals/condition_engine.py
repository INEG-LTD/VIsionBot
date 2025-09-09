"""
Universal condition engine for evaluating logical and environment conditions.

Design goals:
- Deterministic evaluation first (DOM/URL/time), optional AI only for NL → DSL translation.
- JSON DSL to avoid building a heavy string parser initially.
- Lazy, cached fact providers for page, DOM, and system.

Expression (JSON) examples:
- {"and": [ {"contains": [{"var": "env.page.url"}, "dashboard"]}, {"call": {"name": "dom.visible", "args": {"selector": "#submit", "within": "viewport"}}} ]}
- {">=": [ {"call": {"name": "dom.count", "args": {"selector": ".item", "within": "viewport"}}}, 3 ]}
- {"between": [ {"call": {"name": "system.hour"}}, 9, 17 ]}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import datetime
import json
import re

from playwright.sync_api import Page

from .base import Condition, GoalContext, create_environment_condition
from ai_utils import generate_text
from utils.page_utils import PageUtils
from element_detection.overlay_manager import OverlayManager
from element_detection.element_detector import ElementDetector


JsonExpr = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


@dataclass
class EvalOutcome:
    value: bool
    confidence: float = 1.0
    evidence: Dict[str, Any] = None


class FactCache:
    """Per-evaluation cache of facts to avoid repeated DOM calls."""
    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.trace: List[Dict[str, Any]] = []

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value
    
    def add_trace(self, entry: Dict[str, Any]) -> None:
        if len(self.trace) < 200:
            self.trace.append(entry)


class ConditionEngine:
    def __init__(self):
        self.functions: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._register_builtins()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def evaluate(self, expr: JsonExpr, context: GoalContext) -> EvalOutcome:
        cache = FactCache()
        try:
            val = self._eval(expr, context, cache)
            # Normalize to boolean for top-level; non-bool truthiness allowed inside
            return EvalOutcome(
                value=bool(val),
                confidence=1.0,
                evidence={
                    "expr": expr,
                    "url": context.current_state.url,
                    "title": context.current_state.title,
                    "trace": cache.trace,
                },
            )
        except Exception as e:
            print(f"[ConditionEngine] Evaluation error: {e}")
            return EvalOutcome(value=False, confidence=0.0, evidence={"error": str(e), "expr": expr})

    # ---------------------------------------------------------------------
    # Evaluator
    # ---------------------------------------------------------------------
    def _eval(self, expr: JsonExpr, context: GoalContext, cache: FactCache) -> Any:
        # Literals
        if isinstance(expr, (str, int, float, bool)) or expr is None:
            return expr

        # Arrays: evaluate each item
        if isinstance(expr, list):
            return [self._eval(e, context, cache) for e in expr]

        if not isinstance(expr, dict):
            raise ValueError(f"Unsupported expression type: {type(expr)}")

        if not expr:
            return None

        # Single-key operator or function call
        if "var" in expr:
            return self._resolve_var(expr["var"], context, cache)

        if "call" in expr:
            call = expr["call"]
            if not isinstance(call, dict) or "name" not in call:
                raise ValueError("Invalid call expression")
            name = call["name"]
            args = call.get("args", {})
            return self._call_function(name, args, context, cache)

        # Logical operators
        if "and" in expr:
            arr = expr["and"]
            return all(self._eval(e, context, cache) for e in arr)
        if "or" in expr:
            arr = expr["or"]
            return any(self._eval(e, context, cache) for e in arr)
        if "not" in expr:
            return not bool(self._eval(expr["not"], context, cache))

        # Comparison operators (binary)
        for op in ("==", "!=", ">", "<", ">=", "<="):
            if op in expr:
                a, b = expr[op]
                va = self._eval(a, context, cache)
                vb = self._eval(b, context, cache)
                if op == "==":
                    return va == vb
                if op == "!=":
                    return va != vb
                if op == ">":
                    return va > vb
                if op == "<":
                    return va < vb
                if op == ">=":
                    return va >= vb
                if op == "<=":
                    return va <= vb

        # Utility ops
        if "contains" in expr:
            a, b = expr["contains"]
            va = self._eval(a, context, cache)
            vb = self._eval(b, context, cache)
            if isinstance(va, str) and isinstance(vb, str):
                return vb.lower() in va.lower()
            if isinstance(va, list):
                return vb in va
            return False
        if "regex" in expr:
            a, pattern = expr["regex"]
            va = self._eval(a, context, cache)
            if not isinstance(va, str):
                return False
            try:
                return re.search(pattern, va or "") is not None
            except re.error:
                return False
        if "len" in expr:
            v = self._eval(expr["len"], context, cache)
            try:
                return len(v)
            except Exception:
                return 0
        if "between" in expr:
            v, lo, hi = expr["between"]
            vv = self._eval(v, context, cache)
            return (vv >= lo) and (vv <= hi)

        raise ValueError(f"Unknown expression: {expr}")

    # ---------------------------------------------------------------------
    # Providers and built-in calls
    # ---------------------------------------------------------------------
    def _register_builtins(self) -> None:
        self.functions.update({
            # System
            "system.hour": self._fn_system_hour,
            "system.minute": self._fn_system_minute,
            "system.weekday": self._fn_system_weekday,
            "system.weekend": self._fn_system_weekend,
            # Page
            "env.page.url": self._fn_page_url,
            "env.page.title": self._fn_page_title,
            "env.page.viewport_text": self._fn_page_viewport_text,
            "env.page.full_text": self._fn_page_full_text,
            # Page scroll/geometry helpers
            "env.page.scroll_y": self._fn_page_scroll_y,
            "env.page.viewport_height": self._fn_page_viewport_height,
            "env.page.document_height": self._fn_page_document_height,
            "env.page.at_bottom": self._fn_page_at_bottom,
            "env.page.at_top": self._fn_page_at_top,
            # DOM
            "dom.exists": self._fn_dom_exists,
            "dom.visible": self._fn_dom_visible,
            "dom.count": self._fn_dom_count,
            "dom.text": self._fn_dom_text,
            "dom.value": self._fn_dom_value,
            "dom.attr": self._fn_dom_attr,
            # Semantic helpers
            "dom.visible_button": self._fn_dom_visible_button,
            "dom.button_count": self._fn_dom_button_count,
            "dom.visible_link": self._fn_dom_visible_link,
            "dom.link_count": self._fn_dom_link_count,
            "dom.field_filled_by_label": self._fn_dom_field_filled_by_label,
            # AI-assisted helpers (opt-in, slower)
            "ai.visible_button": self._fn_ai_visible_button,
            "ai.is_visible_nl": self._fn_ai_is_visible_nl,
        })

    def _call_function(self, name: str, args: Dict[str, Any], context: GoalContext, cache: FactCache) -> Any:
        fn = self.functions.get(name)
        if not fn:
            raise ValueError(f"Unknown function '{name}'")
        result = fn({"args": args, "context": context, "cache": cache})
        try:
            summary = result
            if isinstance(summary, str) and len(summary) > 100:
                summary = summary[:100] + "…"
            cache.add_trace({"call": name, "args": args, "result": summary})
        except Exception:
            pass
        return result

    # ------------------------- System providers ----------------------------
    def _fn_system_hour(self, p: Dict[str, Any]) -> int:
        return int(datetime.datetime.now().hour)

    def _fn_system_minute(self, p: Dict[str, Any]) -> int:
        return int(datetime.datetime.now().minute)

    def _fn_system_weekday(self, p: Dict[str, Any]) -> bool:
        return datetime.datetime.now().weekday() < 5

    def _fn_system_weekend(self, p: Dict[str, Any]) -> bool:
        return datetime.datetime.now().weekday() >= 5

    # ------------------------- Page providers ------------------------------
    def _page(self, context: GoalContext) -> Optional[Page]:
        return context.page_reference

    def _fn_page_url(self, p: Dict[str, Any]) -> str:
        ctx: GoalContext = p["context"]
        val = ctx.current_state.url or ""
        try:
            p["cache"].add_trace({"var": "env.page.url", "value": val})
        except Exception:
            pass
        return val

    def _fn_page_title(self, p: Dict[str, Any]) -> str:
        ctx: GoalContext = p["context"]
        val = ctx.current_state.title or ""
        try:
            p["cache"].add_trace({"var": "env.page.title", "value": val})
        except Exception:
            pass
        return val

    def _fn_page_full_text(self, p: Dict[str, Any]) -> str:
        ctx: GoalContext = p["context"]
        val = (ctx.current_state.visible_text or "").strip()
        try:
            p["cache"].add_trace({"var": "env.page.full_text", "len": len(val)})
        except Exception:
            pass
        return val

    def _fn_page_scroll_y(self, p: Dict[str, Any]) -> int:
        ctx: GoalContext = p["context"]
        page = self._page(ctx)
        try:
            if page:
                y = int(page.evaluate("() => window.scrollY || 0"))
                p["cache"].add_trace({"var": "env.page.scroll_y", "value": y})
                return y
        except Exception:
            pass
        try:
            y = int(ctx.current_state.scroll_y)
            p["cache"].add_trace({"var": "env.page.scroll_y", "value": y})
            return y
        except Exception:
            return 0

    def _fn_page_viewport_height(self, p: Dict[str, Any]) -> int:
        ctx: GoalContext = p["context"]
        page = self._page(ctx)
        try:
            if page and page.viewport_size:
                h = int(page.viewport_size.get("height") or 0)
                if h:
                    p["cache"].add_trace({"var": "env.page.viewport_height", "value": h})
                    return h
        except Exception:
            pass
        try:
            h = int(ctx.current_state.page_height)
            p["cache"].add_trace({"var": "env.page.viewport_height", "value": h})
            return h
        except Exception:
            return 0

    def _fn_page_document_height(self, p: Dict[str, Any]) -> int:
        ctx: GoalContext = p["context"]
        page = self._page(ctx)
        try:
            if page:
                h = int(page.evaluate("() => (document && document.body && document.body.scrollHeight) || 0"))
                p["cache"].add_trace({"var": "env.page.document_height", "value": h})
                return h
        except Exception:
            pass
        # Fallback to viewport height when DOM not accessible
        try:
            h = int(ctx.current_state.page_height)
            p["cache"].add_trace({"var": "env.page.document_height", "value": h})
            return h
        except Exception:
            return 0

    def _fn_page_at_bottom(self, p: Dict[str, Any]) -> bool:
        ctx: GoalContext = p["context"]
        args: Dict[str, Any] = p.get("args", {})
        threshold = int(args.get("threshold", 100))  # px tolerance
        try:
            page = self._page(ctx)
            if page:
                res = page.evaluate(
                    "({th}) => { const y = window.scrollY||0; const h = window.innerHeight||0; const dh = (document && document.body && document.body.scrollHeight)||0; return (y + h) >= (dh - th); }",
                    {"th": threshold}
                )
                p["cache"].add_trace({"call": "env.page.at_bottom", "args": {"threshold": threshold}, "result": bool(res)})
                return bool(res)
        except Exception:
            pass
        # Fallback using current_state snapshot
        try:
            y = int(ctx.current_state.scroll_y)
            vh = int(ctx.current_state.page_height)
            # Without doc height, assume viewport as doc → can't be at bottom unless y==0. Treat as False conservatively
            dh = vh
            val = (y + vh) >= (dh - threshold)
            p["cache"].add_trace({"call": "env.page.at_bottom", "args": {"threshold": threshold}, "result": bool(val), "mode": "fallback"})
            return bool(val)
        except Exception:
            return False

    def _fn_page_at_top(self, p: Dict[str, Any]) -> bool:
        ctx: GoalContext = p["context"]
        args: Dict[str, Any] = p.get("args", {})
        threshold = int(args.get("threshold", 20))
        try:
            page = self._page(ctx)
            if page:
                y = int(page.evaluate("() => window.scrollY || 0"))
                res = y <= threshold
                p["cache"].add_trace({"call": "env.page.at_top", "args": {"threshold": threshold}, "result": bool(res)})
                return res
        except Exception:
            pass
        try:
            y = int(ctx.current_state.scroll_y)
            res = y <= threshold
            p["cache"].add_trace({"call": "env.page.at_top", "args": {"threshold": threshold}, "result": bool(res), "mode": "fallback"})
            return res
        except Exception:
            return True

    def _fn_page_viewport_text(self, p: Dict[str, Any]) -> str:
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        key = "env.page.viewport_text"
        cached = cache.get(key)
        if cached is not None:
            return cached
        page = self._page(ctx)
        if not page:
            cache.set(key, "")
            return ""

        js_collect = r"""
        () => {
          const normalize = s => String(s || '').replace(/\s+/g, ' ').trim();
          const W = window.innerWidth, H = window.innerHeight;
          function elementAndAncestorsVisible(el) {
            let e = el;
            while (e && e !== document.documentElement) {
              const s = getComputedStyle(e);
              if (s.display === 'none' || s.visibility !== 'visible' || parseFloat(s.opacity || '1') === 0) return false;
              e = e.parentElement;
            }
            return true;
          }
          function rectIntersectsViewport(r) {
            return r.width > 0 && r.height > 0 && r.right > 0 && r.bottom > 0 && r.left < W && r.top < H;
          }
          const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            { acceptNode: (n) => {
                const v = n.nodeValue;
                if (!v || !v.trim()) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
            }}
          );
          let parts = [];
          while (walker.nextNode()) {
            const n = walker.currentNode;
            const p = n.parentElement; if (!p) continue;
            if (!elementAndAncestorsVisible(p)) continue;
            let range;
            try { range = document.createRange(); range.selectNodeContents(n); } catch (e) { continue; }
            const rects = range.getClientRects();
            let inViewport = false;
            for (let i=0;i<rects.length;i++){ if (rectIntersectsViewport(rects[i])) { inViewport = true; break; } }
            if (!inViewport) continue;
            parts.push(normalize(n.nodeValue));
          }
          return parts.join(' ').trim();
        }
        """

        try:
            text_main = page.evaluate(js_collect) or ""
        except Exception:
            text_main = ""

        # Include visible iframes' viewport text
        def _frame_in_viewport(frame) -> bool:
            try:
                fe = frame.frame_element()
                bb = fe.bounding_box() if fe else None
                if not bb:
                    return False
                W = page.viewport_size.get("width", 0) if page.viewport_size else 0
                H = page.viewport_size.get("height", 0) if page.viewport_size else 0
                return not (bb["x"] > W or bb["y"] > H or (bb["x"] + bb["width"]) < 0 or (bb["y"] + bb["height"]) < 0)
            except Exception:
                return False

        texts = [text_main]
        for frame in page.frames:
            if frame == page.main_frame:
                continue
            if not _frame_in_viewport(frame):
                continue
            try:
                t = frame.evaluate(js_collect) or ""
                if t:
                    texts.append(t)
            except Exception:
                continue

        combined = " ".join([t for t in texts if t]).strip()
        try:
            cache.add_trace({"var": "env.page.viewport_text", "len": len(combined)})
        except Exception:
            pass
        cache.set(key, combined)
        return combined

    # ------------------------- DOM providers --------------------------------
    def _fn_dom_exists(self, p: Dict[str, Any]) -> bool:
        return self._dom_eval(p, mode="exists", method="dom.exists")

    def _fn_dom_visible(self, p: Dict[str, Any]) -> bool:
        return self._dom_eval(p, mode="visible", method="dom.visible")

    def _fn_dom_count(self, p: Dict[str, Any]) -> int:
        return self._dom_eval(p, mode="count", method="dom.count")

    def _fn_dom_text(self, p: Dict[str, Any]) -> str:
        return self._dom_eval(p, mode="text", method="dom.text")

    def _fn_dom_value(self, p: Dict[str, Any]) -> str:
        return self._dom_eval(p, mode="value", method="dom.value")

    def _fn_dom_attr(self, p: Dict[str, Any]) -> str:
        return self._dom_eval(p, mode="attr", method="dom.attr")

    def _dom_eval(self, p: Dict[str, Any], mode: str, method: str) -> Any:
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        page = self._page(ctx)
        if not page:
            return 0 if mode == "count" else ("" if mode in ("text", "value", "attr") else False)

        selector = args.get("selector", "")
        within = (args.get("within") or "page").lower()  # 'viewport' or 'page'
        attr_name = args.get("name")
        include_iframes = bool(args.get("iframes", True))

        js = r"""
        ({ selector, within, attrName }) => {
          const W = window.innerWidth, H = window.innerHeight;
          function isVisible(el) {
            const s = getComputedStyle(el);
            if (s.display === 'none' || s.visibility !== 'visible' || parseFloat(s.opacity || '1') === 0) return false;
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) return false;
            if (within === 'viewport') {
              if (r.right <= 0 || r.bottom <= 0 || r.left >= W || r.top >= H) return false;
            }
            return true;
          }
          const nodes = Array.from(document.querySelectorAll(selector));
          const vis = nodes.filter(n => isVisible(n));
          return {
            exists: nodes.length > 0,
            anyVisible: vis.length > 0,
            countAll: nodes.length,
            countVisible: vis.length,
            firstVisibleText: vis[0]?.innerText?.trim() || '',
            firstVisibleValue: (vis[0] && (vis[0].value !== undefined) ? String(vis[0].value) : ''),
            firstVisibleAttr: (vis[0] && attrName) ? (vis[0].getAttribute(attrName) || '') : ''
          };
        }
        """

        def in_viewport(frame) -> bool:
            try:
                fe = frame.frame_element()
                bb = fe.bounding_box() if fe else None
                if not bb:
                    return False
                W = page.viewport_size.get("width", 0) if page.viewport_size else 0
                H = page.viewport_size.get("height", 0) if page.viewport_size else 0
                return not (bb["x"] > W or bb["y"] > H or (bb["x"] + bb["width"]) < 0 or (bb["y"] + bb["height"]) < 0)
            except Exception:
                return False

        def eval_on(frame) -> Dict[str, Any]:
            try:
                return frame.evaluate(js, {"selector": selector, "within": within, "attrName": attr_name}) or {}
            except Exception:
                return {}

        # Main frame first
        agg = eval_on(page.main_frame)

        # Iframes if requested
        if include_iframes:
            for f in page.frames:
                if f == page.main_frame:
                    continue
                if within == "viewport" and not in_viewport(f):
                    continue
                r = eval_on(f)
                # Aggregate counts and any/exists
                agg = {
                    "exists": bool(agg.get("exists") or r.get("exists")),
                    "anyVisible": bool(agg.get("anyVisible") or r.get("anyVisible")),
                    "countAll": int(agg.get("countAll", 0)) + int(r.get("countAll", 0)),
                    "countVisible": int(agg.get("countVisible", 0)) + int(r.get("countVisible", 0)),
                    "firstVisibleText": agg.get("firstVisibleText") or r.get("firstVisibleText") or '',
                    "firstVisibleValue": agg.get("firstVisibleValue") or r.get("firstVisibleValue") or '',
                    "firstVisibleAttr": agg.get("firstVisibleAttr") or r.get("firstVisibleAttr") or ''
                }

        try:
            p["cache"].add_trace({
                "call": method,
                "args": {"selector": selector, "within": within, "name": attr_name},
                "summary": {
                    "exists": bool(agg.get("exists")),
                    "visible": bool(agg.get("anyVisible")),
                    "count_all": int(agg.get("countAll", 0)),
                    "count_visible": int(agg.get("countVisible", 0)),
                }
            })
        except Exception:
            pass

        if mode == "exists":
            return bool(agg.get("exists"))
        if mode == "visible":
            return bool(agg.get("anyVisible"))
        if mode == "count":
            return int(agg.get("countVisible" if within == "viewport" else "countAll", 0))
        if mode == "text":
            return str(agg.get("firstVisibleText", ""))
        if mode == "value":
            return str(agg.get("firstVisibleValue", ""))
        if mode == "attr":
            return str(agg.get("firstVisibleAttr", ""))
        return False

    def _resolve_var(self, path: str, context: GoalContext, cache: FactCache) -> Any:
        # Only expose safe, deterministic vars
        try:
            if path == "env.page.url":
                return self._fn_page_url({"context": context, "args": {}, "cache": cache})
            if path == "env.page.title":
                return self._fn_page_title({"context": context, "args": {}, "cache": cache})
            if path == "env.page.full_text":
                return self._fn_page_full_text({"context": context, "args": {}, "cache": cache})
            if path == "env.page.viewport_text":
                return self._fn_page_viewport_text({"context": context, "args": {}, "cache": cache})
            if path == "system.hour":
                return self._fn_system_hour({"context": context, "args": {}, "cache": cache})
            if path == "system.minute":
                return self._fn_system_minute({"context": context, "args": {}, "cache": cache})
            if path == "system.weekday":
                return self._fn_system_weekday({"context": context, "args": {}, "cache": cache})
            if path == "system.weekend":
                return self._fn_system_weekend({"context": context, "args": {}, "cache": cache})
        except Exception:
            return None
        return None

    # ------------------------- Semantic DOM helpers -------------------------
    def _label_variants(self, label: str, fuzzy: bool) -> List[str]:
        lbl = (label or "").strip().lower()
        variants = [lbl]
        if not fuzzy:
            return variants
        # Basic synonym support for common CTAs
        synonym_map = {
            "submit": ["submit", "send", "apply", "continue", "next", "save", "confirm", "finish", "complete"],
            "login": ["login", "log in", "sign in"],
            "signup": ["signup", "sign up", "register", "create account"],
            "search": ["search", "find", "go"],
            "download": ["download", "save", "get"],
        }
        for k, vals in synonym_map.items():
            if lbl == k:
                variants = list(dict.fromkeys(vals))  # unique, preserve order
                break
        return variants

    def _fn_dom_visible_button(self, p: Dict[str, Any]) -> bool:
        return bool(self._semantic_button_eval(p, want="any", method="dom.visible_button"))

    def _fn_dom_button_count(self, p: Dict[str, Any]) -> int:
        return int(self._semantic_button_eval(p, want="count", method="dom.button_count") or 0)

    def _semantic_button_eval(self, p: Dict[str, Any], want: str, method: str) -> Union[bool, int]:
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        page = self._page(ctx)
        if not page:
            return 0 if want == "count" else False

        label = args.get("label", "")
        within = (args.get("within") or "viewport").lower()
        include_iframes = bool(args.get("iframes", True))
        fuzzy = bool(args.get("fuzzy", True))
        variants = self._label_variants(label, fuzzy)

        js = r"""
        ({ variants, within }) => {
          const W = window.innerWidth, H = window.innerHeight;
          function isVisible(el) {
            const s = getComputedStyle(el);
            if (s.display === 'none' || s.visibility !== 'visible' || parseFloat(s.opacity || '1') === 0) return false;
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) return false;
            if (within === 'viewport') {
              if (r.right <= 0 || r.bottom <= 0 || r.left >= W || r.top >= H) return false;
            }
            return true;
          }
          function labelFor(el) {
            const parts = [];
            const txt = (el.innerText || '').trim(); if (txt) parts.push(txt);
            const val = (el.value !== undefined ? String(el.value) : ''); if (val) parts.push(val);
            const aria = el.getAttribute && (el.getAttribute('aria-label') || ''); if (aria) parts.push(aria);
            const title = el.getAttribute && (el.getAttribute('title') || ''); if (title) parts.push(title);
            return parts.join(' ').replace(/\s+/g, ' ').trim().toLowerCase();
          }
          function matches(lbl, variants) {
            if (!lbl) return false;
            for (const v of variants) {
              if (lbl.includes(String(v).toLowerCase())) return true;
            }
            return false;
          }
          const nodes = Array.from(document.querySelectorAll('button, input[type="submit"], input[type="button"], [role="button"]'));
          let visibleMatches = [];
          for (const n of nodes) {
            if (!isVisible(n)) continue;
            const lbl = labelFor(n);
            if (matches(lbl, variants)) visibleMatches.push(n);
          }
          return {
            any: visibleMatches.length > 0,
            count: visibleMatches.length
          };
        }
        """

        def in_viewport(frame) -> bool:
            try:
                fe = frame.frame_element()
                bb = fe.bounding_box() if fe else None
                if not bb:
                    return False
                W = page.viewport_size.get("width", 0) if page.viewport_size else 0
                H = page.viewport_size.get("height", 0) if page.viewport_size else 0
                return not (bb["x"] > W or bb["y"] > H or (bb["x"] + bb["width"]) < 0 or (bb["y"] + bb["height"]) < 0)
            except Exception:
                return False

        def eval_on(frame) -> Dict[str, Any]:
            try:
                return frame.evaluate(js, {"variants": variants, "within": within}) or {"any": False, "count": 0}
            except Exception:
                return {"any": False, "count": 0}

        agg = eval_on(page.main_frame)
        if include_iframes:
            for f in page.frames:
                if f == page.main_frame:
                    continue
                if within == "viewport" and not in_viewport(f):
                    continue
                r = eval_on(f)
                agg = {"any": bool(agg.get("any") or r.get("any")), "count": int(agg.get("count", 0)) + int(r.get("count", 0))}
        result = agg.get(want, 0 if want == "count" else False)
        try:
            p["cache"].add_trace({"call": method, "args": {"label": label, "within": within, "fuzzy": fuzzy}, "result": result})
        except Exception:
            pass
        return result

    # Link semantics ---------------------------------------------------------
    def _fn_dom_visible_link(self, p: Dict[str, Any]) -> bool:
        return bool(self._semantic_link_eval(p, want="any", method="dom.visible_link"))

    def _fn_dom_link_count(self, p: Dict[str, Any]) -> int:
        return int(self._semantic_link_eval(p, want="count", method="dom.link_count") or 0)

    def _semantic_link_eval(self, p: Dict[str, Any], want: str, method: str) -> Union[bool, int]:
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        page = self._page(ctx)
        if not page:
            return 0 if want == "count" else False

        label = args.get("label", "")
        within = (args.get("within") or "viewport").lower()
        include_iframes = bool(args.get("iframes", True))

        js = r"""
        ({ label, within }) => {
          const W = window.innerWidth, H = window.innerHeight;
          function isVisible(el) {
            const s = getComputedStyle(el);
            if (s.display === 'none' || s.visibility !== 'visible' || parseFloat(s.opacity || '1') === 0) return false;
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) return false;
            if (within === 'viewport') { if (r.right <= 0 || r.bottom <= 0 || r.left >= W || r.top >= H) return false; }
            return true;
          }
          function labelFor(el) {
            const parts = [];
            const txt = (el.innerText || '').trim(); if (txt) parts.push(txt);
            const aria = el.getAttribute && (el.getAttribute('aria-label') || ''); if (aria) parts.push(aria);
            const title = el.getAttribute && (el.getAttribute('title') || ''); if (title) parts.push(title);
            return parts.join(' ').replace(/\s+/g, ' ').trim().toLowerCase();
          }
          const q = String(label || '').toLowerCase();
          const nodes = Array.from(document.querySelectorAll('a, [role="link"]'));
          let visibleMatches = [];
          for (const n of nodes) { if (!isVisible(n)) continue; if (labelFor(n).includes(q)) visibleMatches.push(n); }
          return { any: visibleMatches.length > 0, count: visibleMatches.length };
        }
        """

        def in_viewport(frame) -> bool:
            try:
                fe = frame.frame_element()
                bb = fe.bounding_box() if fe else None
                if not bb:
                    return False
                W = page.viewport_size.get("width", 0) if page.viewport_size else 0
                H = page.viewport_size.get("height", 0) if page.viewport_size else 0
                return not (bb["x"] > W or bb["y"] > H or (bb["x"] + bb["width"]) < 0 or (bb["y"] + bb["height"]) < 0)
            except Exception:
                return False

        def eval_on(frame) -> Dict[str, Any]:
            try:
                return frame.evaluate(js, {"label": label, "within": within}) or {"any": False, "count": 0}
            except Exception:
                return {"any": False, "count": 0}

        agg = eval_on(page.main_frame)
        if include_iframes:
            for f in page.frames:
                if f == page.main_frame: continue
                if within == "viewport" and not in_viewport(f): continue
                r = eval_on(f)
                agg = {"any": bool(agg.get("any") or r.get("any")), "count": int(agg.get("count", 0)) + int(r.get("count", 0))}

        result = agg.get(want, 0 if want == "count" else False)
        try:
            p["cache"].add_trace({"call": method, "args": {"label": label, "within": within}, "result": result})
        except Exception:
            pass
        return result

    # Field filled by label ---------------------------------------------------
    def _fn_dom_field_filled_by_label(self, p: Dict[str, Any]) -> bool:
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        page = self._page(ctx)
        if not page:
            return False
        label = args.get("label", "")
        within = (args.get("within") or "viewport").lower()
        include_iframes = bool(args.get("iframes", True))

        js = """
        ({ label, within }) => {
          const W = window.innerWidth, H = window.innerHeight;
          function isVisible(el) {
            const s = getComputedStyle(el);
            if (s.display === 'none' || s.visibility !== 'visible' || parseFloat(s.opacity || '1') === 0) return false;
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) return false;
            if (within === 'viewport') { if (r.right <= 0 || r.bottom <= 0 || r.left >= W || r.top >= H) return false; }
            return true;
          }
          function byId(id) { try { return document.getElementById(id); } catch(e) { return null; } }
          function isFilled(el) {
            if (!el) return false;
            const tag = (el.tagName||'').toLowerCase();
            const type = (el.getAttribute && (el.getAttribute('type')||'').toLowerCase()) || '';
            if (tag === 'select') return el.selectedIndex >= 0 && !!(el.value||'').trim();
            if (tag === 'textarea') return !!(el.value||'').trim();
            if (tag === 'input') {
              if (['checkbox','radio'].includes(type)) return !!el.checked;
              return !!(el.value||'').trim();
            }
            return false;
          }
          const q = String(label||'').toLowerCase();
          // Labels
          const labels = Array.from(document.querySelectorAll('label')).filter(l => isVisible(l) && (l.innerText||'').toLowerCase().includes(q));
          for (const lab of labels) {
            let target = null;
            const forId = lab.getAttribute('for');
            if (forId) target = byId(forId);
            if (!target) target = lab.querySelector('input,select,textarea');
            if (target && isVisible(target) && isFilled(target)) return true;
          }
          // aria-labelledby
          const controls = Array.from(document.querySelectorAll('input,select,textarea'));
          for (const c of controls) {
            const ids = (c.getAttribute('aria-labelledby')||'').split(/\s+/).filter(Boolean);
            if (!ids.length) continue;
            let text = '';
            for (const id of ids) { const el = byId(id); if (el) text += ' ' + (el.innerText||''); }
            if (!text) continue;
            if (text.toLowerCase().includes(q) && isVisible(c) && isFilled(c)) return true;
          }
          return false;
        }
        """

        def in_viewport(frame) -> bool:
            try:
                fe = frame.frame_element()
                bb = fe.bounding_box() if fe else None
                if not bb:
                    return False
                W = page.viewport_size.get("width", 0) if page.viewport_size else 0
                H = page.viewport_size.get("height", 0) if page.viewport_size else 0
                return not (bb["x"] > W or bb["y"] > H or (bb["x"] + bb["width"]) < 0 or (bb["y"] + bb["height"]) < 0)
            except Exception:
                return False

        def eval_on(frame) -> bool:
            try:
                return bool(frame.evaluate(js, {"label": label, "within": within}))
            except Exception:
                return False

        result = eval_on(page.main_frame)
        if include_iframes and not result:
            for f in page.frames:
                if f == page.main_frame: continue
                if within == "viewport" and not in_viewport(f): continue
                if eval_on(f):
                    result = True
                    break

        try:
            p["cache"].add_trace({"call": "dom.field_filled_by_label", "args": {"label": label, "within": within}, "result": result})
        except Exception:
            pass
        return result

    # ------------------------- AI-assisted helpers --------------------------
    def _fn_ai_visible_button(self, p: Dict[str, Any]) -> bool:
        """Use AI (overlay-based detector) to find a visible button by label when deterministic lookup fails.

        Flow:
        1) Try deterministic dom.visible_button first.
        2) Create numbered overlays; capture screenshot; ask vision model for relevant overlay(s).
        3) Map overlay -> DOM via data-automation-overlay-index; verify visibility + label match; cleanup overlays.
        """
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return False

        label = args.get("label", "")
        within = (args.get("within") or "viewport").lower()
        iframes = bool(args.get("iframes", True))
        fuzzy = bool(args.get("fuzzy", True))

        # Basic cache to avoid repeated calls on same query within same evaluation
        cache_key = f"ai.visible_button::{label.lower()}::{within}::{fuzzy}"
        cached = cache.get(cache_key)
        if cached is not None:
            return bool(cached)

        # Step 1: quick deterministic check
        if self._fn_dom_visible_button({"args": {"label": label, "within": within, "iframes": iframes, "fuzzy": fuzzy}, "context": ctx, "cache": p.get("cache") }):
            return True

        overlays = None
        try:
            # Step 2: prepare overlays + screenshot
            page_utils = PageUtils(page)
            page_info = page_utils.get_page_info()
            overlays = OverlayManager(page)
            overlay_data = overlays.create_numbered_overlays(page_info)
            if not overlay_data:
                return False
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)

            det = ElementDetector(model_name="gemini-2.5-flash-lite")
            goal_desc = f"Find a visible button for label '{label}'"
            result = det.detect_elements_with_overlays(
                goal_description=goal_desc,
                additional_context="",
                screenshot=screenshot,
                element_data=overlay_data,
                page_info=page_info,
            )
            if not result or not getattr(result, 'elements', None):
                return False

            # Step 3: map overlays to DOM and verify
            js_verify = r"""
            ({ overlayIndex, within, label, fuzzy }) => {
              const W = window.innerWidth, H = window.innerHeight;
              function isVisible(el) {
                const s = getComputedStyle(el);
                if (s.display === 'none' || s.visibility !== 'visible' || parseFloat(s.opacity || '1') === 0) return false;
                const r = el.getBoundingClientRect();
                if (r.width === 0 || r.height === 0) return false;
                if (within === 'viewport') {
                  if (r.right <= 0 || r.bottom <= 0 || r.left >= W || r.top >= H) return false;
                }
                return true;
              }
              function labelFor(el) {
                const parts = [];
                const txt = (el.innerText || '').trim(); if (txt) parts.push(txt);
                const val = (el.value !== undefined ? String(el.value) : ''); if (val) parts.push(val);
                const aria = el.getAttribute && (el.getAttribute('aria-label') || ''); if (aria) parts.push(aria);
                const title = el.getAttribute && (el.getAttribute('title') || ''); if (title) parts.push(title);
                return parts.join(' ').replace(/\s+/g, ' ').trim().toLowerCase();
              }
              function matches(lbl, label, fuzzy) {
                const q = String(label || '').toLowerCase();
                if (!fuzzy) return lbl.includes(q);
                const synonyms = { submit: ['submit','send','apply','continue','next','save','confirm','finish','complete'] };
                const pool = synonyms[q] || [q];
                return pool.some(v => lbl.includes(v));
              }
              const sel = `[data-automation-overlay-index="${overlayIndex}"]`;
              const el = document.querySelector(sel);
              if (!el) return false;
              if (!isVisible(el)) return false;
              const lbl = labelFor(el);
              return matches(lbl, label, fuzzy);
            }
            """

            for elem in result.elements:
                try:
                    overlay_idx = getattr(elem, 'overlay_number', None)
                    elem_type = getattr(elem, 'element_type', '') or ''
                    is_clickable = bool(getattr(elem, 'is_clickable', False))
                    if not overlay_idx:
                        continue
                    if 'button' not in elem_type.lower() and not is_clickable:
                        continue
                    ok = page.evaluate(js_verify, {"overlayIndex": overlay_idx, "within": within, "label": label, "fuzzy": fuzzy})
                    if ok:
                        try:
                            cache.add_trace({
                                "call": "ai.visible_button",
                                "args": {"label": label, "within": within, "fuzzy": fuzzy},
                                "overlay": overlay_idx,
                                "element_type": elem_type,
                                "used_ai": True
                            })
                        except Exception:
                            pass
                        cache.set(cache_key, True)
                        return True
                except Exception:
                    continue
        finally:
            try:
                if overlays:
                    overlays.remove_overlays()
            except Exception:
                pass

        cache.set(cache_key, False)
        return False

    def _fn_ai_is_visible_nl(self, p: Dict[str, Any]) -> bool:
        """Text visibility check with exact (quoted) vs fuzzy (unquoted) matching.

        Args:
            query: Text to search for. If in quotes, performs exact text search.
                  If not in quotes, performs fuzzy/semantic AI search.
            within: "viewport" | "page" (default: viewport)
            iframes: bool (currently not used here)

        Returns True if the text is visible now.
        """
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return False

        query = (args.get("query") or "").strip()
        if not query:
            return False

        # Check if query is in quotes (exact match) or not (fuzzy match)
        is_exact_match = query.startswith('"') and query.endswith('"')
        if is_exact_match:
            # Remove quotes for exact text search
            search_text = query[1:-1]
            return self._exact_text_search(search_text, ctx, cache, args.get("within", "viewport"))
        else:
            # Use AI for fuzzy/semantic matching
            return self._fuzzy_text_search(query, ctx, cache, args.get("within", "viewport"))

    def _exact_text_search(self, search_text: str, ctx: GoalContext, cache: FactCache, within: str) -> bool:
        """Perform deterministic exact text search in viewport."""
        cache_key = f"exact_text_search::{search_text.lower()}"
        cached = cache.get(cache_key)
        if cached is not None:
            return bool(cached)

        try:
            # Get viewport text
            viewport_text = self._fn_page_viewport_text({"context": ctx, "args": {}, "cache": cache})
            
            # Perform case-insensitive exact search
            found = search_text.lower() in viewport_text.lower()
            
            cache.add_trace({"call": "exact_text_search", "args": {"query": search_text, "within": within}, "result": found})
            cache.set(cache_key, found)
            return found
        except Exception as e:
            cache.add_trace({"call": "exact_text_search", "args": {"query": search_text, "within": within}, "error": str(e)})
            cache.set(cache_key, False)
            return False

    def _fuzzy_text_search(self, query: str, ctx: GoalContext, cache: FactCache, within: str) -> bool:
        """Perform AI-assisted fuzzy/semantic text search."""
        cache_key = f"fuzzy_text_search::{query.lower()}"
        cached = cache.get(cache_key)
        if cached is not None:
            return bool(cached)

        page = self._page(ctx)
        if not page:
            return False

        overlays = None
        try:
            page_utils = PageUtils(page)
            page_info = page_utils.get_page_info()
            overlays = OverlayManager(page)
            overlay_data = overlays.create_numbered_overlays(page_info) or []

            # Collect a short viewport text excerpt for context
            try:
                viewport_text = self._fn_page_viewport_text({"context": ctx, "args": {}, "cache": cache})
            except Exception:
                viewport_text = ""

            # Build compact overlay summaries
            summaries = []
            for el in overlay_data[:60]:
                try:
                    summaries.append({
                        "i": el.get('index'),
                        "tag": (el.get('tagName') or '')[:20],
                        "role": (el.get('role') or '')[:20],
                        "name": (el.get('name') or '')[:40],
                        "aria": (el.get('ariaLabel') or '')[:80],
                        "id": (el.get('id') or '')[:60],
                        "class": (el.get('className') or '')[:100],
                        "text": (el.get('textContent') or '')[:140],
                    })
                except Exception:
                    continue

            from ai_utils import generate_text
            import json as _json

            prompt = (
                "Decide if text matching the query is visible in the viewport.\n"
                "Look for semantic matches, not just exact text matches.\n"
                "Return only JSON: {\"visible\": true|false}.\n\n"
                f"Query: {query}\n\n"
                f"Overlays (first {len(summaries)}):\n" + _json.dumps(summaries)[:5500] + "\n\n"
                "Viewport text excerpt (first 1000 chars):\n" + (viewport_text[:1000] or '')
            )

            result = generate_text(
                prompt=prompt,
                system_prompt="Answer with strict JSON only as {\"visible\": true|false}.",
                reasoning_level="minimal",
                model="gpt-5-nano",
            )
            try:
                parsed = _json.loads((result or "").strip())
                vis = bool(parsed.get("visible", False))
                cache.add_trace({"call": "fuzzy_text_search", "args": {"query": query, "within": within}, "result": vis})
                cache.set(cache_key, vis)
                return vis
            except Exception:
                cache.add_trace({"call": "fuzzy_text_search", "args": {"query": query, "within": within}, "result": "non-json"})
                cache.set(cache_key, False)
                return False
        except Exception as e:
            try:
                cache.add_trace({"call": "fuzzy_text_search", "args": {"query": query, "within": within}, "error": str(e)})
            except Exception:
                pass
            cache.set(cache_key, False)
            return False
        finally:
            try:
                if overlays:
                    overlays.remove_overlays()
            except Exception:
                pass


def get_default_engine() -> ConditionEngine:
    return ConditionEngine()


def create_predicate_condition(expression: JsonExpr, description: Optional[str] = None) -> Condition:
    """Create a Condition from a JSON expression using the default engine."""
    if description is None:
        description = "Predicate condition"
    engine = get_default_engine()

    def evaluator(context: GoalContext) -> bool:
        outcome = engine.evaluate(expression, context)
        # Emit concise evidence for debugging
        try:
            trace = outcome.evidence.get("trace", []) if outcome.evidence else []
            if trace:
                print(f"[Predicate] {description} => {outcome.value}")
                for t in trace[-5:]:
                    print(f"  - {t}")
        except Exception:
            pass
        return outcome.value

    return create_environment_condition(description, evaluator)


def compile_nl_to_expr(condition_text: str) -> Optional[JsonExpr]:
    """Use AI to compile natural language into the JSON DSL expression.

    The AI is used only as a translator; factual evaluation remains deterministic
    in the engine/providers.
    """
    system = (
        "You translate natural language conditions into a strict JSON expression DSL. "
        "No explanations. Output only valid JSON."
    )
    dsl_guide = (
        "DSL operators: and/or/not, ==, !=, >, <, >=, <=, contains, regex, len, between, var, call.\n"
        "Use {\"var\": \"env.page.url\"} to access facts.\n"
        "Available calls (call.name):\n"
        " - env.page.url, env.page.title, env.page.viewport_text, env.page.full_text (no args)\n"
        " - env.page.scroll_y(), env.page.viewport_height(), env.page.document_height()\n"
        " - env.page.at_bottom(threshold=px), env.page.at_top(threshold=px)\n"
        " - dom.exists(selector, within=page|viewport, iframes=true)\n"
        " - dom.visible(selector, within=page|viewport, iframes=true)\n"
        " - dom.count(selector, within=page|viewport, iframes=true)\n"
        " - dom.text(selector, within=page|viewport, iframes=true)\n"
        " - dom.value(selector, within=page|viewport, iframes=true)\n"
        " - dom.attr(selector, name, within=page|viewport, iframes=true)\n"
        " - dom.visible_button(label, within=page|viewport, iframes=true, fuzzy=true)\n"
        " - dom.button_count(label, within=page|viewport, iframes=true, fuzzy=true)\n"
        " - dom.visible_link(label, within=page|viewport, iframes=true)\n"
        " - dom.link_count(label, within=page|viewport, iframes=true)\n"
        " - dom.field_filled_by_label(label, within=page|viewport, iframes=true)\n"
        " - ai.is_visible_nl(query, within=page|viewport)  # Text search: quoted=exact match, unquoted=fuzzy match\n"
        " - system.hour(), system.minute(), system.weekday(), system.weekend()\n"
        "Examples:\n"
        "1) Page shows Welcome and URL has dashboard:\n"
        "{\"and\": [ {\"contains\": [ {\"call\": {\"name\": \"env.page.url\"}}, \"dashboard\"]}, {\"contains\": [ {\"call\": {\"name\": \"env.page.viewport_text\"}}, \"Welcome\"]} ]}\n"
        "2) Exactly 3 visible .item elements in viewport:\n"
        "{\"==\": [ {\"call\": {\"name\": \"dom.count\", \"args\": {\"selector\": \".item\", \"within\": \"viewport\"}}}, 3 ]}\n"
        "3) Submit button visible:\n"
        "{\"call\": {\"name\": \"dom.visible_button\", \"args\": {\"label\": \"submit\", \"within\": \"viewport\"}}}\n"
        "4) Cookie banner visible (no selector known):\n"
        "{\"call\": {\"name\": \"ai.is_visible_nl\", \"args\": {\"query\": \"cookie banner\", \"within\": \"viewport\"}}}\n"
        "5) Newsletter modal visible:\n"
        "{\"call\": {\"name\": \"ai.is_visible_nl\", \"args\": {\"query\": \"newsletter modal\"}}}\n"
        "6) At bottom of the page:\n"
        "{\"call\": {\"name\": \"env.page.at_bottom\"}}\n"
        "7) Not at bottom of the page:\n"
        "{\"not\": {\"call\": {\"name\": \"env.page.at_bottom\", \"args\": {\"threshold\": 120}}}}\n"
        "8) Next button visible (pagination):\n"
        "{\"or\": [ {\"call\": {\"name\": \"dom.visible_button\", \"args\": {\"label\": \"next\", \"within\": \"viewport\"}}}, {\"call\": {\"name\": \"dom.visible_link\", \"args\": {\"label\": \"next\", \"within\": \"viewport\"}}}, {\"call\": {\"name\": \"dom.visible\", \"args\": {\"selector\": \"a[rel=next], button[aria-label*=next] \" , \"within\": \"viewport\"}}} ]}\n"
        "9) Can't see exact text (quoted = exact match, unquoted = fuzzy match):\n"
        "{\"not\": {\"call\": {\"name\": \"ai.is_visible_nl\", \"args\": {\"query\": \"\\\"Proven L4 Autonomy\\\"\", \"within\": \"viewport\"}}}}\n"
        "10) Can't see fuzzy text (semantic matching):\n"
        "{\"not\": {\"call\": {\"name\": \"ai.is_visible_nl\", \"args\": {\"query\": \"autonomous vehicle technology\", \"within\": \"viewport\"}}}}\n"
    )
    prompt = f"Condition: {condition_text}\nReturn only the JSON expression."
    try:
        out = generate_text(prompt=prompt, system_prompt=dsl_guide + "\n" + system, reasoning_level="minimal", model="gpt-5-nano")
        if not out:
            return None
        # Extract JSON
        text = out.strip()
        # Handle accidental code fences
        if text.startswith("```"):
            text = text.strip('`')
            # Remove possible language hint
            text = text.split('\n', 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        result = json.loads(text)
        print(f"[ConditionEngine] Compiled '{condition_text}' to: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        print(f"[ConditionEngine] NL→DSL compile error: {e}")
        return None
