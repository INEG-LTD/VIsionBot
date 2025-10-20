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
from typing import Any, Dict, List, Optional, Union, Callable
import datetime
import json
import re

from playwright.sync_api import Page

from .base import Condition, GoalContext, create_environment_condition
from ai_utils import generate_text, rewrite_condition_to_question, answer_question_with_vision, get_default_model
from utils.page_utils import PageUtils
from element_detection.overlay_manager import OverlayManager
from element_detection.element_detector import ElementDetector
from utils.vision_resolver import visible_from_overlays, dom_text_visible


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
            # Vision deterministic helpers
            "vision.visible": self._fn_vision_visible,
        })

    def _call_function(self, name: str, args: Dict[str, Any], context: GoalContext, cache: FactCache) -> Any:
        # Check if this is a page function that should use deterministic evaluation
        if name.startswith("env.page."):
            return self._call_page_function(name, args, context, cache)
        
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
    
    def _call_page_function(self, name: str, args: Dict[str, Any], context: GoalContext, cache: FactCache) -> Any:
        """Call deterministic page functions instead of AI-based ones."""
        try:
            from .page_functions import get_page_function_registry
            registry = get_page_function_registry()
            func = registry.get_function(name)
            
            if func:
                page = self._page(context)
                if not page:
                    return self._get_fallback_value(name, context)
                
                result = func(page, args)
                cache.add_trace({"call": name, "args": args, "result": result, "method": "javascript"})
                return result
            else:
                # Fallback to original AI-based function if not found in registry
                fn = self.functions.get(name)
                if fn:
                    result = fn({"args": args, "context": context, "cache": cache})
                    cache.add_trace({"call": name, "args": args, "result": result, "method": "ai_fallback"})
                    return result
                else:
                    raise ValueError(f"Unknown page function '{name}'")
        except Exception as e:
            cache.add_trace({"call": name, "args": args, "error": str(e), "method": "error"})
            return self._get_fallback_value(name, context)
    
    def _get_fallback_value(self, name: str, context: GoalContext) -> Any:
        """Get fallback values for page functions when page is not available."""
        if name == "env.page.url":
            return context.current_state.url or ""
        elif name == "env.page.title":
            return context.current_state.title or ""
        elif name in ["env.page.scroll_y", "env.page.scroll_x"]:
            return 0
        elif name in ["env.page.viewport_height", "env.page.document_height"]:
            return int(context.current_state.page_height) if context.current_state.page_height else 0
        elif name in ["env.page.at_bottom", "env.page.at_top"]:
            return False
        elif name == "env.page.scroll_percentage":
            return 0.0
        else:
            return None

    # ------------------------- System providers ----------------------------
    def _fn_system_hour(self, p: Dict[str, Any]) -> int:
        """Use LLM to determine current hour."""
        cache: FactCache = p["cache"]
        
        try:
            # Use LLM to determine current hour
            prompt = "What is the current hour of the day (0-23)? Answer with just a number."
            result = generate_text(
                prompt=prompt,
                system_prompt="You are a time assistant. Answer with just the current hour as a number between 0-23.",
            )
            
            if result:
                hour = int(str(result).strip())
                cache.add_trace({"call": "system.hour", "result": hour, "method": "llm"})
                return max(0, min(23, hour))
            
            # Fallback to system time
            hour = int(datetime.datetime.now().hour)
            cache.add_trace({"call": "system.hour", "result": hour, "method": "fallback"})
            return hour
        except Exception as e:
            cache.add_trace({"call": "system.hour", "error": str(e)})
            return int(datetime.datetime.now().hour)

    def _fn_system_minute(self, p: Dict[str, Any]) -> int:
        """Use LLM to determine current minute."""
        cache: FactCache = p["cache"]
        
        try:
            # Use LLM to determine current minute
            prompt = "What is the current minute of the hour (0-59)? Answer with just a number."
            result = generate_text(
                prompt=prompt,
                system_prompt="You are a time assistant. Answer with just the current minute as a number between 0-59.",
            )
            
            if result:
                minute = int(str(result).strip())
                cache.add_trace({"call": "system.minute", "result": minute, "method": "llm"})
                return max(0, min(59, minute))
            
            # Fallback to system time
            minute = int(datetime.datetime.now().minute)
            cache.add_trace({"call": "system.minute", "result": minute, "method": "fallback"})
            return minute
        except Exception as e:
            cache.add_trace({"call": "system.minute", "error": str(e)})
            return int(datetime.datetime.now().minute)

    def _fn_system_weekday(self, p: Dict[str, Any]) -> bool:
        """Use LLM to determine if it's a weekday."""
        cache: FactCache = p["cache"]
        
        try:
            # Use LLM to determine if it's a weekday
            prompt = "Is today a weekday (Monday through Friday)? Answer with just 'yes' or 'no'."
            result = generate_text(
                prompt=prompt,
                system_prompt="You are a time assistant. Answer with just 'yes' or 'no' for whether today is a weekday.",
            )
            
            if result:
                is_weekday = str(result).lower().strip() in ['yes', 'true', '1']
                cache.add_trace({"call": "system.weekday", "result": is_weekday, "method": "llm"})
                return is_weekday
            
            # Fallback to system time
            is_weekday = datetime.datetime.now().weekday() < 5
            cache.add_trace({"call": "system.weekday", "result": is_weekday, "method": "fallback"})
            return is_weekday
        except Exception as e:
            cache.add_trace({"call": "system.weekday", "error": str(e)})
            return datetime.datetime.now().weekday() < 5

    def _fn_system_weekend(self, p: Dict[str, Any]) -> bool:
        """Use LLM to determine if it's a weekend."""
        cache: FactCache = p["cache"]
        
        try:
            # Use LLM to determine if it's a weekend
            prompt = "Is today a weekend (Saturday or Sunday)? Answer with just 'yes' or 'no'."
            result = generate_text(
                prompt=prompt,
                system_prompt="You are a time assistant. Answer with just 'yes' or 'no' for whether today is a weekend.",
            )
            
            if result:
                is_weekend = str(result).lower().strip() in ['yes', 'true', '1']
                cache.add_trace({"call": "system.weekend", "result": is_weekend, "method": "llm"})
                return is_weekend
            
            # Fallback to system time
            is_weekend = datetime.datetime.now().weekday() >= 5
            cache.add_trace({"call": "system.weekend", "result": is_weekend, "method": "fallback"})
            return is_weekend
        except Exception as e:
            cache.add_trace({"call": "system.weekend", "error": str(e)})
            return datetime.datetime.now().weekday() >= 5

    # ------------------------- Page providers ------------------------------
    def _page(self, context: GoalContext) -> Optional[Page]:
        return context.page_reference

    def _fn_page_url(self, p: Dict[str, Any]) -> str:
        """Use LLM to extract URL from page context."""
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        
        # Use LLM to analyze page and extract URL
        try:
            page = self._page(ctx)
            if not page:
                return ctx.current_state.url or ""
            
            # Get page context for LLM analysis
            page_utils = PageUtils(page)
            page_info = page_utils.get_page_info()
            
            # Use LLM to extract URL from page context
            prompt = f"Analyze this page and extract the current URL. Page title: {page_info.get('title', '')}. Return only the URL."
            result = generate_text(
                prompt=prompt,
                system_prompt="Extract the URL from the page information. Return only the URL.",
            )
            
            url = str(result or "").strip()
            if not url:
                url = ctx.current_state.url or ""
            
            cache.add_trace({"var": "env.page.url", "value": url, "method": "llm"})
            return url
        except Exception as e:
            cache.add_trace({"var": "env.page.url", "error": str(e)})
            return ctx.current_state.url or ""

    def _fn_page_title(self, p: Dict[str, Any]) -> str:
        """Use LLM to extract title from page context."""
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        
        # Use LLM to analyze page and extract title
        try:
            page = self._page(ctx)
            if not page:
                return ctx.current_state.title or ""
            
            # Get page context for LLM analysis
            page_utils = PageUtils(page)
            page_info = page_utils.get_page_info()
            
            # Use LLM to extract title from page context
            prompt = f"Analyze this page and extract the page title. Page content preview: {page_info.get('content', '')[:500]}. Return only the title."
            result = generate_text(
                prompt=prompt,
                system_prompt="Extract the page title from the page information. Return only the title.",
            )
            
            title = str(result or "").strip()
            if not title:
                title = ctx.current_state.title or ""
            
            cache.add_trace({"var": "env.page.title", "value": title, "method": "llm"})
            return title
        except Exception as e:
            cache.add_trace({"var": "env.page.title", "error": str(e)})
            return ctx.current_state.title or ""

    def _fn_page_full_text(self, p: Dict[str, Any]) -> str:
        """Use vision/LLM to extract full page text."""
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        
        try:
            page = self._page(ctx)
            if not page:
                return (ctx.current_state.visible_text or "").strip()
            
            # Use vision to extract all visible text from the page
            question = "What is all the visible text content on this page? Include text from all sections, headers, paragraphs, and UI elements."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=True)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            text = str(vision_answer or "").strip()
            
            if not text:
                text = (ctx.current_state.visible_text or "").strip()
            
            cache.add_trace({"var": "env.page.full_text", "len": len(text), "method": "vision"})
            return text
        except Exception as e:
            cache.add_trace({"var": "env.page.full_text", "error": str(e)})
            return (ctx.current_state.visible_text or "").strip()

    def _fn_page_scroll_y(self, p: Dict[str, Any]) -> int:
        """Use vision/LLM to determine scroll position."""
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        
        try:
            page = self._page(ctx)
            if not page:
                return int(ctx.current_state.scroll_y) if ctx.current_state.scroll_y else 0
            
            # Use vision to analyze scroll position
            question = "What is the current vertical scroll position of this page? Answer with just a number representing pixels scrolled from the top."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            if vision_answer is not None:
                try:
                    y = int(str(vision_answer).strip())
                    cache.add_trace({"var": "env.page.scroll_y", "value": y, "method": "vision"})
                    return max(0, y)
                except ValueError:
                    pass
            
            # Fallback to context state
            y = int(ctx.current_state.scroll_y) if ctx.current_state.scroll_y else 0
            cache.add_trace({"var": "env.page.scroll_y", "value": y, "method": "fallback"})
            return y
        except Exception as e:
            cache.add_trace({"var": "env.page.scroll_y", "error": str(e)})
            return int(ctx.current_state.scroll_y) if ctx.current_state.scroll_y else 0

    def _fn_page_viewport_height(self, p: Dict[str, Any]) -> int:
        """Use vision/LLM to determine viewport height."""
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        
        try:
            page = self._page(ctx)
            if not page:
                return int(ctx.current_state.page_height) if ctx.current_state.page_height else 0
            
            # Use vision to analyze viewport height
            question = "What is the height of the visible viewport area of this page? Answer with just a number representing pixels."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            if vision_answer is not None:
                try:
                    h = int(str(vision_answer).strip())
                    cache.add_trace({"var": "env.page.viewport_height", "value": h, "method": "vision"})
                    return max(0, h)
                except ValueError:
                    pass
            
            # Fallback to context state
            h = int(ctx.current_state.page_height) if ctx.current_state.page_height else 0
            cache.add_trace({"var": "env.page.viewport_height", "value": h, "method": "fallback"})
            return h
        except Exception as e:
            cache.add_trace({"var": "env.page.viewport_height", "error": str(e)})
            return int(ctx.current_state.page_height) if ctx.current_state.page_height else 0

    def _fn_page_document_height(self, p: Dict[str, Any]) -> int:
        """Use vision/LLM to determine document height."""
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        
        try:
            page = self._page(ctx)
            if not page:
                return int(ctx.current_state.page_height) if ctx.current_state.page_height else 0
            
            # Use vision to analyze document height
            question = "What is the total height of the entire document/page content? Answer with just a number representing pixels."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=True)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            if vision_answer is not None:
                try:
                    h = int(str(vision_answer).strip())
                    cache.add_trace({"var": "env.page.document_height", "value": h, "method": "vision"})
                    return max(0, h)
                except ValueError:
                    pass
            
            # Fallback to context state
            h = int(ctx.current_state.page_height) if ctx.current_state.page_height else 0
            cache.add_trace({"var": "env.page.document_height", "value": h, "method": "fallback"})
            return h
        except Exception as e:
            cache.add_trace({"var": "env.page.document_height", "error": str(e)})
            return int(ctx.current_state.page_height) if ctx.current_state.page_height else 0

    def _fn_page_at_bottom(self, p: Dict[str, Any]) -> bool:
        """Use vision/LLM to determine if page is at bottom."""
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        args: Dict[str, Any] = p.get("args", {})
        threshold = int(args.get("threshold", 100))
        
        try:
            page = self._page(ctx)
            if not page:
                return False
            
            # Use vision to analyze if page is at bottom
            question = f"Is this page scrolled to the bottom? Consider that we might be within {threshold} pixels of the bottom as 'at bottom'. Answer with just 'yes' or 'no'."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            if vision_answer is not None:
                result = str(vision_answer).lower().strip() in ['yes', 'true', '1']
                cache.add_trace({"call": "env.page.at_bottom", "args": {"threshold": threshold}, "result": result, "method": "vision"})
                return result
            
            # Fallback using context state
            y = int(ctx.current_state.scroll_y) if ctx.current_state.scroll_y else 0
            vh = int(ctx.current_state.page_height) if ctx.current_state.page_height else 0
            dh = vh  # Assume viewport as doc height
            val = (y + vh) >= (dh - threshold)
            cache.add_trace({"call": "env.page.at_bottom", "args": {"threshold": threshold}, "result": val, "method": "fallback"})
            return val
        except Exception as e:
            cache.add_trace({"call": "env.page.at_bottom", "args": {"threshold": threshold}, "error": str(e)})
            return False

    def _fn_page_at_top(self, p: Dict[str, Any]) -> bool:
        """Use vision/LLM to determine if page is at top."""
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        args: Dict[str, Any] = p.get("args", {})
        threshold = int(args.get("threshold", 20))
        
        try:
            page = self._page(ctx)
            if not page:
                return True
            
            # Use vision to analyze if page is at top
            question = f"Is this page scrolled to the top? Consider that we might be within {threshold} pixels of the top as 'at top'. Answer with just 'yes' or 'no'."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            if vision_answer is not None:
                result = str(vision_answer).lower().strip() in ['yes', 'true', '1']
                cache.add_trace({"call": "env.page.at_top", "args": {"threshold": threshold}, "result": result, "method": "vision"})
                return result
            
            # Fallback using context state
            y = int(ctx.current_state.scroll_y) if ctx.current_state.scroll_y else 0
            result = y <= threshold
            cache.add_trace({"call": "env.page.at_top", "args": {"threshold": threshold}, "result": result, "method": "fallback"})
            return result
        except Exception as e:
            cache.add_trace({"call": "env.page.at_top", "args": {"threshold": threshold}, "error": str(e)})
            return True

    def _fn_page_viewport_text(self, p: Dict[str, Any]) -> str:
        """Use vision/LLM to extract viewport text."""
        ctx: GoalContext = p["context"]
        cache: FactCache = p["cache"]
        key = "env.page.viewport_text"
        cached = cache.get(key)
        if cached is not None:
            return cached
        
        try:
            page = self._page(ctx)
            if not page:
                cache.set(key, "")
                return ""
            
            # Use vision to extract viewport text
            question = "What is all the visible text content in the current viewport? Include text from headers, paragraphs, buttons, links, and all other visible elements."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            text = str(vision_answer or "").strip()
            
            cache.add_trace({"var": "env.page.viewport_text", "len": len(text), "method": "vision"})
            cache.set(key, text)
            return text
        except Exception as e:
            cache.add_trace({"var": "env.page.viewport_text", "error": str(e)})
            cache.set(key, "")
            return ""

    # ------------------------- DOM providers --------------------------------
    def _fn_dom_exists(self, p: Dict[str, Any]) -> bool:
        """Use vision/LLM to check if elements matching a description exist."""
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return False

        description = args.get("description", "") or args.get("selector", "")
        within = (args.get("within") or "viewport").lower()
        if not description:
            return False

        # Use vision-based element detection with natural language description
        try:
            question = f"Are there any elements that match this description visible in the {within}: '{description}'? Answer with just 'yes' or 'no'."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            # Use vision to answer the question
            vision_answer = answer_question_with_vision(question, screenshot)
            result = str(vision_answer).lower().strip() in ['yes', 'true', '1']
            
            cache.add_trace({"call": "dom.exists", "args": {"description": description, "within": within}, "result": result, "method": "vision"})
            return result
        except Exception as e:
            cache.add_trace({"call": "dom.exists", "args": {"description": description, "within": within}, "error": str(e)})
            return False

    def _fn_dom_visible(self, p: Dict[str, Any]) -> bool:
        """Use vision/LLM to check if elements matching a description are visible."""
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return False

        description = args.get("description", "") or args.get("selector", "")
        within = (args.get("within") or "viewport").lower()
        if not description:
            return False

        # Use vision-based visibility check with natural language description
        try:
            question = f"Are there any visible elements that match this description in the {within}: '{description}'? Answer with just 'yes' or 'no'."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            result = str(vision_answer).lower().strip() in ['yes', 'true', '1']
            
            cache.add_trace({"call": "dom.visible", "args": {"description": description, "within": within}, "result": result, "method": "vision"})
            return result
        except Exception as e:
            cache.add_trace({"call": "dom.visible", "args": {"description": description, "within": within}, "error": str(e)})
            return False

    def _fn_dom_count(self, p: Dict[str, Any]) -> int:
        """Use vision/LLM to count elements matching a description."""
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return 0

        description = args.get("description", "") or args.get("selector", "")
        within = (args.get("within") or "viewport").lower()
        if not description:
            return 0

        # Use vision-based counting with natural language description
        try:
            question = f"How many elements that match this description are visible in the {within}: '{description}'? Answer with just a number."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            if vision_answer is not None:
                try:
                    result = int(str(vision_answer).strip())
                    cache.add_trace({"call": "dom.count", "args": {"description": description, "within": within}, "result": result, "method": "vision"})
                    return max(0, result)
                except ValueError:
                    pass
            
            cache.add_trace({"call": "dom.count", "args": {"description": description, "within": within}, "result": 0, "method": "vision_fallback"})
            return 0
        except Exception as e:
            cache.add_trace({"call": "dom.count", "args": {"description": description, "within": within}, "error": str(e)})
            return 0

    def _fn_dom_text(self, p: Dict[str, Any]) -> str:
        """Use vision/LLM to extract text from elements matching a description."""
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return ""

        description = args.get("description", "") or args.get("selector", "")
        within = (args.get("within") or "viewport").lower()
        if not description:
            return ""

        # Use vision-based text extraction with natural language description
        try:
            question = f"What is the text content of the first visible element that matches this description in the {within}: '{description}'?"
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            result = str(vision_answer or "").strip()
            
            cache.add_trace({"call": "dom.text", "args": {"description": description, "within": within}, "result": result, "method": "vision"})
            return result
        except Exception as e:
            cache.add_trace({"call": "dom.text", "args": {"description": description, "within": within}, "error": str(e)})
            return ""

    def _fn_dom_value(self, p: Dict[str, Any]) -> str:
        """Use vision/LLM to extract value from input elements matching a description."""
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return ""

        description = args.get("description", "") or args.get("selector", "")
        within = (args.get("within") or "viewport").lower()
        if not description:
            return ""

        # Use vision-based value extraction with natural language description
        try:
            question = f"What is the value of the first visible input field that matches this description in the {within}: '{description}'?"
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            result = str(vision_answer or "").strip()
            
            cache.add_trace({"call": "dom.value", "args": {"description": description, "within": within}, "result": result, "method": "vision"})
            return result
        except Exception as e:
            cache.add_trace({"call": "dom.value", "args": {"description": description, "within": within}, "error": str(e)})
            return ""

    def _fn_dom_attr(self, p: Dict[str, Any]) -> str:
        """Use vision/LLM to extract attribute from elements matching a description."""
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return ""

        description = args.get("description", "") or args.get("selector", "")
        attr_name = args.get("name", "")
        within = (args.get("within") or "viewport").lower()
        if not description or not attr_name:
            return ""

        # Use vision-based attribute extraction with natural language description
        try:
            question = f"What is the value of the '{attr_name}' attribute of the first visible element that matches this description in the {within}: '{description}'?"
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            result = str(vision_answer or "").strip()
            
            cache.add_trace({"call": "dom.attr", "args": {"description": description, "name": attr_name, "within": within}, "result": result, "method": "vision"})
            return result
        except Exception as e:
            cache.add_trace({"call": "dom.attr", "args": {"description": description, "name": attr_name, "within": within}, "error": str(e)})
            return ""


    def _resolve_var(self, path: str, context: GoalContext, cache: FactCache) -> Any:
        # Use LLM/vision-based variable resolution
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
        """Use vision/LLM to evaluate semantic button presence."""
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return 0 if want == "count" else False

        label = args.get("label", "")
        within = (args.get("within") or "viewport").lower()
        fuzzy = bool(args.get("fuzzy", True))
        
        if not label:
            return 0 if want == "count" else False

        try:
            # Use vision to detect buttons with the specified label
            if want == "count":
                question = f"How many buttons with the label '{label}' are visible in the {within}? Answer with just a number."
            else:
                question = f"Are there any buttons with the label '{label}' visible in the {within}? Answer with just 'yes' or 'no'."
            
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            vision_answer = answer_question_with_vision(question, screenshot)
            
            if want == "count":
                try:
                    result = int(str(vision_answer).strip())
                    cache.add_trace({"call": method, "args": {"label": label, "within": within, "fuzzy": fuzzy}, "result": result, "method": "vision"})
                    return max(0, result)
                except ValueError:
                    cache.add_trace({"call": method, "args": {"label": label, "within": within, "fuzzy": fuzzy}, "result": 0, "method": "vision_fallback"})
                    return 0
            else:
                result = str(vision_answer).lower().strip() in ['yes', 'true', '1']
                cache.add_trace({"call": method, "args": {"label": label, "within": within, "fuzzy": fuzzy}, "result": result, "method": "vision"})
                return result
        except Exception as e:
            cache.add_trace({"call": method, "args": {"label": label, "within": within, "fuzzy": fuzzy}, "error": str(e)})
            return 0 if want == "count" else False

    # Link semantics ---------------------------------------------------------
    def _fn_dom_visible_link(self, p: Dict[str, Any]) -> bool:
        return bool(self._semantic_link_eval(p, want="any", method="dom.visible_link"))

    def _fn_dom_link_count(self, p: Dict[str, Any]) -> int:
        return int(self._semantic_link_eval(p, want="count", method="dom.link_count") or 0)

    def _semantic_link_eval(self, p: Dict[str, Any], want: str, method: str) -> Union[bool, int]:
        """Use vision/LLM to evaluate semantic link presence."""
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return 0 if want == "count" else False

        label = args.get("label", "")
        within = (args.get("within") or "viewport").lower()
        
        if not label:
            return 0 if want == "count" else False

        try:
            # Use vision to detect links with the specified label
            if want == "count":
                question = f"How many links with the label '{label}' are visible in the {within}? Answer with just a number."
            else:
                question = f"Are there any links with the label '{label}' visible in the {within}? Answer with just 'yes' or 'no'."
            
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            vision_answer = answer_question_with_vision(question, screenshot)
            
            if want == "count":
                try:
                    result = int(str(vision_answer).strip())
                    cache.add_trace({"call": method, "args": {"label": label, "within": within}, "result": result, "method": "vision"})
                    return max(0, result)
                except ValueError:
                    cache.add_trace({"call": method, "args": {"label": label, "within": within}, "result": 0, "method": "vision_fallback"})
                    return 0
            else:
                result = str(vision_answer).lower().strip() in ['yes', 'true', '1']
                cache.add_trace({"call": method, "args": {"label": label, "within": within}, "result": result, "method": "vision"})
                return result
        except Exception as e:
            cache.add_trace({"call": method, "args": {"label": label, "within": within}, "error": str(e)})
            return 0 if want == "count" else False

    # Field filled by label ---------------------------------------------------
    def _fn_dom_field_filled_by_label(self, p: Dict[str, Any]) -> bool:
        """Use vision/LLM to check if a field is filled by label."""
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return False
        
        label = args.get("label", "")
        within = (args.get("within") or "viewport").lower()
        
        if not label:
            return False

        try:
            # Use vision to check if field is filled
            question = f"Is there a form field with the label '{label}' that is filled/has content in the {within}? Answer with just 'yes' or 'no'."
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
            
            vision_answer = answer_question_with_vision(question, screenshot)
            result = str(vision_answer).lower().strip() in ['yes', 'true', '1']
            
            cache.add_trace({"call": "dom.field_filled_by_label", "args": {"label": label, "within": within}, "result": result, "method": "vision"})
            return result
        except Exception as e:
            cache.add_trace({"call": "dom.field_filled_by_label", "args": {"label": label, "within": within}, "error": str(e)})
            return False

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
        screenshot = None
        try:
            # Step 2: prepare overlays + screenshot
            page_utils = PageUtils(page)
            page_info = page_utils.get_page_info()
            overlays = OverlayManager(page)
            overlay_data = overlays.create_numbered_overlays(page_info, mode="interactive")
            if not overlay_data:
                return False
            # Quick deterministic overlay label match before invoking vision model
            try:
                lbl = (label or '').lower()
                if lbl:
                    for el in overlay_data:
                        text = ((el.get('textContent') or '') + ' ' + (el.get('ariaLabel') or '')).lower()
                        if lbl in text:
                            p["cache"].add_trace({"call": "ai.visible_button.overlay_precheck", "args": {"label": label}, "overlay": el.get('index')})
                            return True
            except Exception:
                pass
            screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)

            det = ElementDetector(model_name=get_default_model())
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

        # Attempt vision QA rewrite if overlays/DOM checks failed
        question = rewrite_condition_to_question(f"Is there a visible button for '{label}'?")
        if not screenshot:
            try:
                screenshot = page.screenshot(type="jpeg", quality=40, full_page=False)
            except Exception:
                screenshot = None

        if screenshot:
            vision_answer = answer_question_with_vision(question, screenshot)
            if vision_answer is not None:
                try:
                    cache.add_trace({
                        "call": "ai.visible_button.vision_qa",
                        "question": question,
                        "result": vision_answer,
                    })
                except Exception:
                    pass
                cache.set(cache_key, bool(vision_answer))
                return bool(vision_answer)

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
            # Prefer deterministic vision checks before AI fuzzy search
            try:
                if dom_text_visible(page, query):
                    cache.add_trace({"call": "vision.dom.visible", "args": {"query": query}, "result": True})
                    return True
            except Exception:
                pass
            overlays = None
            try:
                page_utils = PageUtils(page)
                page_info = page_utils.get_page_info()
                overlays = OverlayManager(page)
                overlay_data = overlays.create_numbered_overlays(page_info, mode="interactive") or []
                if overlay_data and visible_from_overlays(query, overlay_data, mode="click", interpretation_mode="semantic"):
                    cache.add_trace({"call": "vision.overlay.visible", "args": {"query": query}, "result": True})
                    return True
            except Exception:
                pass
            finally:
                try:
                    if overlays:
                        overlays.remove_overlays()
                except Exception:
                    pass

            # Attempt vision QA: rewrite condition to question and ask model using screenshot
            question = rewrite_condition_to_question(query)
            screenshot = ctx.current_state.screenshot if ctx and ctx.current_state else None
            if not screenshot:
                try:
                    screenshot = page.screenshot(type="jpeg", quality=40, full_page=False)
                except Exception:
                    screenshot = None

            if screenshot:
                vision_answer = answer_question_with_vision(question, screenshot)
                if vision_answer is not None:
                    try:
                        cache.add_trace({
                            "call": "ai.is_visible_nl.vision_qa",
                            "question": question,
                            "result": vision_answer,
                        })
                    except Exception:
                        pass
                    return bool(vision_answer)

            # Fallback to AI fuzzy/semantic text-only search
            return self._fuzzy_text_search(query, ctx, cache, args.get("within", "viewport"))

    # ------------------------- Vision helpers ---------------------------------
    def _fn_vision_visible(self, p: Dict[str, Any]) -> bool:
        """Deterministically check if a semantic region/text is visible using vision heuristics.

        Args:
            query: string to resolve (e.g., "cookie banner", "login modal")
            within: "viewport" | "page" (default: viewport) — currently informational for DOM check.
        """
        args = p.get("args", {})
        ctx: GoalContext = p.get("context")
        cache: FactCache = p.get("cache")
        page = self._page(ctx)
        if not page:
            return False

        query = (args.get("query") or "").strip()
        within = (args.get("within") or "viewport").lower()
        if not query:
            return False

        try:
            if dom_text_visible(page, query):
                cache.add_trace({"call": "vision.visible", "args": {"query": query, "within": within}, "mode": "dom", "result": True})
                return True
        except Exception:
            pass

        overlays = None
        try:
            page_utils = PageUtils(page)
            page_info = page_utils.get_page_info()
            overlays = OverlayManager(page)
            overlay_data = overlays.create_numbered_overlays(page_info, mode="interactive") or []
            if overlay_data and visible_from_overlays(query, overlay_data, interpretation_mode="semantic"):
                cache.add_trace({"call": "vision.visible", "args": {"query": query, "within": within}, "mode": "overlay", "result": True})
                return True
        except Exception:
            pass
        finally:
            try:
                if overlays:
                    overlays.remove_overlays()
            except Exception:
                pass

        cache.add_trace({"call": "vision.visible", "args": {"query": query, "within": within}, "result": False})
        return False

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
            overlay_data = overlays.create_numbered_overlays(page_info, mode="interactive") or []

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
        " - env.page.url, env.page.title (no args) - JavaScript-based\n"
        " - env.page.scroll_y(), env.page.scroll_x(), env.page.scroll_percentage() - JavaScript-based\n"
        " - env.page.viewport_width(), env.page.viewport_height(), env.page.viewport_size() - JavaScript-based\n"
        " - env.page.document_width(), env.page.document_height(), env.page.document_size() - JavaScript-based\n"
        " - env.page.at_bottom(threshold=px), env.page.at_top(threshold=px) - JavaScript-based\n"
        " - env.page.can_scroll_down(), env.page.can_scroll_up() - JavaScript-based\n"
        " - env.page.viewport_text, env.page.full_text (no args) - Vision-based (AI fallback)\n"
        " - dom.exists(description, within=page|viewport) - Vision-based, use natural language descriptions\n"
        " - dom.visible(description, within=page|viewport) - Vision-based, use natural language descriptions\n"
        " - dom.count(description, within=page|viewport) - Vision-based, use natural language descriptions\n"
        " - dom.text(description, within=page|viewport) - Vision-based, use natural language descriptions\n"
        " - dom.value(description, within=page|viewport) - Vision-based, use natural language descriptions\n"
        " - dom.attr(description, name, within=page|viewport) - Vision-based, use natural language descriptions\n"
        " - dom.visible_button(label, within=page|viewport, iframes=true, fuzzy=true) - Vision-based\n"
        " - dom.button_count(label, within=page|viewport, iframes=true, fuzzy=true) - Vision-based\n"
        " - dom.visible_link(label, within=page|viewport, iframes=true) - Vision-based\n"
        " - dom.link_count(label, within=page|viewport, iframes=true) - Vision-based\n"
        " - dom.field_filled_by_label(label, within=page|viewport, iframes=true) - Vision-based\n"
        " - vision.visible(query, within=page|viewport)  # Vision semantic visibility check\n"
        " - ai.is_visible_nl(query, within=page|viewport)  # Text search: quoted=exact match, unquoted=fuzzy match\n"
        " - system.hour(), system.minute(), system.weekday(), system.weekend() - LLM-based\n"
        "Examples:\n"
        "1) Page shows Welcome and URL has dashboard:\n"
        "{\"and\": [ {\"contains\": [ {\"call\": {\"name\": \"env.page.url\"}}, \"dashboard\"]}, {\"contains\": [ {\"call\": {\"name\": \"env.page.viewport_text\"}}, \"Welcome\"]} ]}\n"
        "2) Exactly 3 visible product items in viewport:\n"
        "{\"==\": [ {\"call\": {\"name\": \"dom.count\", \"args\": {\"description\": \"product items\", \"within\": \"viewport\"}}}, 3 ]}\n"
        "3) Submit button visible:\n"
        "{\"call\": {\"name\": \"dom.visible_button\", \"args\": {\"label\": \"submit\", \"within\": \"viewport\"}}}\n"
        "4) Red button with 'Save' text visible:\n"
        "{\"call\": {\"name\": \"dom.visible\", \"args\": {\"description\": \"red button with Save text\", \"within\": \"viewport\"}}}\n"
        "5) Login form field filled:\n"
        "{\"call\": {\"name\": \"dom.field_filled_by_label\", \"args\": {\"label\": \"username\", \"within\": \"viewport\"}}}\n"
        "6) Cookie banner visible:\n"
        "{\"call\": {\"name\": \"ai.is_visible_nl\", \"args\": {\"query\": \"cookie banner\", \"within\": \"viewport\"}}}\n"
        "7) Newsletter modal visible:\n"
        "{\"call\": {\"name\": \"ai.is_visible_nl\", \"args\": {\"query\": \"newsletter modal\"}}}\n"
        "8) At bottom of the page:\n"
        "{\"call\": {\"name\": \"env.page.at_bottom\"}}\n"
        "9) Not at bottom of the page:\n"
        "{\"not\": {\"call\": {\"name\": \"env.page.at_bottom\", \"args\": {\"threshold\": 120}}}}\n"
        "10) Scrolled more than 50% down:\n"
        "{\">\": [ {\"call\": {\"name\": \"env.page.scroll_percentage\"}}, 0.5 ]}\n"
        "11) Can scroll down:\n"
        "{\"call\": {\"name\": \"env.page.can_scroll_down\"}}\n"
        "12) Viewport is wide enough (>= 1200px):\n"
        "{\">=\": [ {\"call\": {\"name\": \"env.page.viewport_width\"}}, 1200 ]}\n"
        "13) Next button visible (pagination):\n"
        "{\"or\": [ {\"call\": {\"name\": \"dom.visible_button\", \"args\": {\"label\": \"next\", \"within\": \"viewport\"}}}, {\"call\": {\"name\": \"dom.visible_link\", \"args\": {\"label\": \"next\", \"within\": \"viewport\"}}}, {\"call\": {\"name\": \"dom.visible\", \"args\": {\"description\": \"next page button or link\", \"within\": \"viewport\"}}} ]}\n"
        "14) Can't see exact text (quoted = exact match, unquoted = fuzzy match):\n"
        "{\"not\": {\"call\": {\"name\": \"ai.is_visible_nl\", \"args\": {\"query\": \"\\\"Proven L4 Autonomy\\\"\", \"within\": \"viewport\"}}}}\n"
        "15) Can't see fuzzy text (semantic matching):\n"
        "{\"not\": {\"call\": {\"name\": \"ai.is_visible_nl\", \"args\": {\"query\": \"autonomous vehicle technology\", \"within\": \"viewport\"}}}}\n"
    )
    prompt = f"Condition: {condition_text}\nReturn only the JSON expression."
    try:
        out = generate_text(prompt=prompt, system_prompt=dsl_guide + "\n" + system)
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
        
        # Debug: print the raw text before parsing
        print(f"[ConditionEngine] Raw AI output: '{text}'")
        
        # Try to fix common JSON issues
        # Remove any trailing commas before closing braces/brackets
        import re
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Try to extract JSON from text that might have extra content
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        result = json.loads(text)
        print(f"[ConditionEngine] Compiled '{condition_text}' to: {json.dumps(result, indent=2)}")
        return result
    except json.JSONDecodeError as e:
        print(f"[ConditionEngine] JSON parse error: {e}")
        print(f"[ConditionEngine] Raw text that failed to parse: '{text}'")
        # Try to return a simple fallback condition
        return {"call": {"name": "ai.is_visible_nl", "args": {"query": condition_text, "within": "viewport"}}}
    except Exception as e:
        print(f"[ConditionEngine] NL→DSL compile error: {e}")
        return None
