"""
VisionCoordinator (Structured Output): Drive Playwright purely by viewport coordinates using a vision model that returns a **Pydantic BaseModel**.

What changed vs previous version
- Uses a **structured JSON schema** defined with Pydantic `BaseModel` (no jsonschema dependency required).
- Integrates directly with your `generate_model_gpt_with_cost(...)` function to get **parsed objects + cost**.
- Still sends **viewport-only** screenshots and a strict coordinate contract (CSS px, origin at viewport top-left).
- Optional grid overlay on the screenshot for visual coordinate guidance.

Dependencies
- playwright (sync API)
- pillow  # for grid overlay (pip install pillow)
- pydantic>=1.10 (works with v2 as well)

This module avoids selectors entirely.
"""
from __future__ import annotations

from enum import Enum
import hashlib
import io
from dataclasses import dataclass
import json
from typing import Annotated, Any, Dict, List, Mapping, Optional, Tuple, Literal, Union
import re
from PIL import Image, ImageDraw  # type: ignore
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from playwright.sync_api import Page
from ai_utils import generate_model, generate_text
from cookie_handler import CookieHandler
from text_utils import TextUtils
from form_utils import FormUtils
from navigation_utils import NavigationUtils
from plan_utils import PlanUtils
import time
import math
# ---------------------------------------------------------------------------
# Pydantic models for **structured output** from the vision model
# ---------------------------------------------------------------------------

ActionLiteral = Literal[
    "click", "double_click", "right_click", "move", "drag",
    "type", "press", "scroll", "wait"
]

class SectionIn(Enum):
    HEADER = "header"
    FOOTER = "footer"
    SIDEBAR = "sidebar"
    CONTENT = "content"
    MODAL = "modal"
    NAVIGATION = "navigation"
    OTHER = "other"

class DetectedItem(BaseModel):
    """Represents a detected prominent item in the image"""
    description: str = Field(description="Human-readable description of the item")
    viewport_data: ViewportMeta = Field(description="Viewport data of the item")
    box_2d: List[int] = Field(description="Bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000")
    section_in: SectionIn = Field(description="Section the item is in")
    confidence: float = Field(description="Detection confidence 0.0-1.0", ge=0.0, le=1.0)
    triggers_file_input: bool = Field(description="Whether this item triggers a file input dialog")
    item_type: str = Field(description="Type of item (button, link, input, text, image, etc.)")
    input_type: str = Field(description="Type of input (text, email, password, upload, radio, checkbox, select, not an input)")
    value: str = Field(default="", description="Value of the item. If it is an input, it is the value of the input.")
    clickable: bool = Field(default=False, description="Whether this item is clickable/interactive")
    surrounding_context: str = Field(default="", description="Context surrounding the item. Things like the text, links, images etc. around the item.")
    in_dialog: bool = Field(default=False, description="Whether this item is in a dialog/modal")
    is_behind_modal: bool = Field(default=False, description="Whether this item is behind a modal. If there is a modal/dialog open, this item is behind the modal/dialog.")
    viewport_data: ViewportMeta = Field(description="Viewport data of the item")
    
    @field_validator("box_2d")
    @classmethod
    def validate_box(cls, v):
        if len(v) != 4:
            raise ValueError("box_2d must have exactly 4 elements [ymin, xmin, ymax, xmax]")
        ymin, xmin, ymax, xmax = v
        if not (0 <= ymin <= 1000 and 0 <= xmin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmax <= 1000):
            raise ValueError("All box coordinates must be between 0 and 1000")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError("Invalid box: min coordinates must be less than max coordinates")
        return v

class DetectionResult(BaseModel):
    """Result of detecting all prominent items in the image"""
    items: List[DetectedItem] = Field(default_factory=list, description="All detected prominent items")
    image_dimensions: List[int] = Field(description="Image dimensions [width, height] in pixels")
    detection_confidence: float = Field(description="Overall detection confidence", ge=0.0, le=1.0)

class Step(BaseModel):
    action: ActionLiteral
    # For actions that target detected items
    action_note: str = Field(default="", description="A note about the action to be taken")
    target_item_index: Optional[int] = Field(default=None, description="Index of target item from detection results. Start from 0.")
    target_item_confidence: Optional[float] = Field(default=None, description="Confidence of the target item. 0.0-1.0")
    # Direct coordinates (fallback)
    x: Optional[int] = Field(default=None, description="Viewport X in CSS px")
    y: Optional[int] = Field(default=None, description="Viewport Y in CSS px")
    x2: Optional[int] = Field(default=None, description="End X for drag")
    y2: Optional[int] = Field(default=None, description="End Y for drag")
    button: Optional[Literal["left", "right", "middle"]] = "left"
    text: Optional[str] = None            # for `type`
    key: Optional[str] = None             # for `press`
    ms: Optional[int] = Field(default=None, ge=0)  # for `wait`
    to_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # for `scroll`
    
    
    @field_validator("x", "y", "x2", "y2", mode="before")
    @classmethod
    def _intify(cls, v):
        if v is None:
            return v
        try:
            return int(v)
        except Exception:
            return v

class ModelPlan(BaseModel):
    detection_result: DetectionResult
    steps: List[Step] = Field(default_factory=list)
    reason: Optional[str] = None
    confidence: Optional[float] = None

class TaskDecomposition(BaseModel):
    task: str
    sub_tasks: List[str]


# =====================================
# Core stop/page descriptors & policies
# =====================================
class StopWhen(BaseModel):
    headings: List[str] = Field(default_factory=list)
    url_contains: List[str] = Field(default_factory=list)
    title_contains: List[str] = Field(default_factory=list)
    body_contains: List[str] = Field(default_factory=list)
    aria_active_tab: List[str] = Field(default_factory=list)
    zero_required_fields_remaining: bool = False
    match_mode: Literal["any", "all"] = "any"

    # NEW — robustness knobs
    hostname_in: List[str] = Field(default_factory=list)     # e.g., ["are.na"]
    path_contains: List[str] = Field(default_factory=list)   # e.g., ["/about"]
    require_url_change: bool = False                         # ensure we actually navigated
    min_strong_signals: int = 1                              # require ≥N “strong” hits when specified



class CompletionPolicy(BaseModel):
    """Cross-site submission/confirmation detector config."""
    intent: str = "job application submitted successfully"
    success_phrases: List[str] = Field(default_factory=lambda: [
        "application submitted", "thanks for applying", "your application has been submitted",
        "we have received your application", "submission complete", "successfully submitted",
        "thank you for your application", "thank you for applying"
    ])
    negative_phrases: List[str] = Field(default_factory=lambda: [
        "required field", "please complete", "error", "fix the following"
    ])
    url_success_substrings: List[str] = Field(default_factory=lambda: [
        "thank-you", "thankyou", "submitted", "confirmation", "success"
    ])
    require_no_submit_button: bool = True
    require_form_not_present: bool = False
    require_no_required_fields_remaining: bool = False
    use_screenshot_llm_check: bool = True
    use_email_confirmation_check: bool = False
    min_score: float = 0.6

# ======================
# Discriminated goal set
# ======================
class GoalKind(str, Enum):
    PAGE_REACHED = "page_reached"               # e.g., reach Sprint 5 page
    SUBMISSION_CONFIRMED = "submission_confirmed"
    FORM_COMPLETED = "form_completed"
    LOGIN_COMPLETED = "login_completed"
    ANY_OF = "any_of"                           # meta-goal: any child goal suffices
    ALL_OF = "all_of"                           # meta-goal: all child goals required


class GoalBase(BaseModel):
    type: GoalKind


class PageReachedGoal(GoalBase):
    type: Literal[GoalKind.PAGE_REACHED] = Field(default=GoalKind.PAGE_REACHED)
    stop_when: StopWhen


class SubmissionConfirmedGoal(GoalBase):
    type: Literal[GoalKind.SUBMISSION_CONFIRMED] = Field(default=GoalKind.SUBMISSION_CONFIRMED)
    policy: CompletionPolicy = Field(default_factory=CompletionPolicy)


class FormCompletedGoal(GoalBase):
    type: Literal[GoalKind.FORM_COMPLETED] = Field(default=GoalKind.FORM_COMPLETED)
    require_no_required_fields_remaining: bool = True
    require_no_visible_form: bool = False


class LoginCompletedGoal(GoalBase):
    type: Literal[GoalKind.LOGIN_COMPLETED] = Field(default=GoalKind.LOGIN_COMPLETED)
    logout_phrases: List[str] = Field(default_factory=lambda: ["log out", "sign out", "logout"])
    require_no_login_form: bool = True


class AnyOfGoal(GoalBase):
    type: Literal[GoalKind.ANY_OF] = Field(default=GoalKind.ANY_OF)
    goals: List["Goal"] = Field(default_factory=list)


class AllOfGoal(GoalBase):
    type: Literal[GoalKind.ALL_OF] = Field(default=GoalKind.ALL_OF)
    goals: List["Goal"] = Field(default_factory=list)


Goal = Annotated[
    Union[
        PageReachedGoal,
        SubmissionConfirmedGoal,
        FormCompletedGoal,
        LoginCompletedGoal,
        AnyOfGoal,
        AllOfGoal,
    ],
    Field(discriminator="type"),
]

# ========================
# Execution policies / hints
# ========================
class ProgressPolicy(BaseModel):
    prefer_cta_text: List[str] = Field(default_factory=list)
    avoid_cta_text: List[str] = Field(default_factory=list)
    fill_only: str = "required"          # "required" | "all"
    login_allowed: bool = True


class UploadPolicy(BaseModel):
    allow_pdf: bool = True
    allow_docx: bool = True
    allow_images: bool = True
    max_mb: int = 25
    prefer: List[str] = Field(default_factory=list)
    match_by: List[str] = Field(default_factory=list)
    if_multiple_uploads_required: str = "first"


class Assets(BaseModel):
    resume_path: Optional[str] = None
    cover_letter_path: Optional[str] = None
    upload_policy: UploadPolicy = Field(default_factory=UploadPolicy)


class PrivacyPolicy(BaseModel):
    pii_budget: str = "minimal"
    redact_logs: bool = True


class SynthesisPolicy(BaseModel):
    strategy: str = "safe_defaults"
    style: str = "neutral"
    max_chars: int = 200


class Policies(BaseModel):
    destructive_actions: str = "block"   # "block" | "allow"
    allow_submit: bool = False
    captcha: str = "stop"                # "stop" | "attempt" | "ignore"
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    synthesis: SynthesisPolicy = Field(default_factory=SynthesisPolicy)
    completion: Optional[CompletionPolicy] = None  # optional override for submission goals


class AdapterHints(BaseModel):
    progress_buttons: List[str] = Field(default_factory=list)
    submit_buttons: List[str] = Field(default_factory=list)
    required_markers: List[str] = Field(default_factory=list)
    stepper_labels: List[str] = Field(default_factory=list)
    upload_labels: List[str] = Field(default_factory=list)


class AdapterDetect(BaseModel):
    url_contains: List[str] = Field(default_factory=list)
    headings: List[str] = Field(default_factory=list)


class Adapter(BaseModel):
    name: str
    detect: AdapterDetect = Field(default_factory=AdapterDetect)
    hints: AdapterHints = Field(default_factory=AdapterHints)

class Auth(BaseModel):
    """Optional creds used for login flows when allowed."""
    username: Optional[str] = None
    password: Optional[str] = None
    email: Optional[str] = None
    otp: Optional[str] = None
    extra: Dict[str, str] = Field(default_factory=dict)  # anything site-specific

class UXHints(BaseModel):
    """Top-level hints & policies passed to your executor."""
    model_config = ConfigDict(extra="ignore")

    # New canonical field: a list of goals (supports composites)
    goals: List[Goal] = Field(default_factory=list)

    goal_progress: Optional[ProgressPolicy] = None
    policies: Policies = Field(default_factory=Policies)
    profile: Dict[str, Any] = Field(default_factory=dict)
    assets: Assets = Field(default_factory=Assets)
    adapters: List[Adapter] = Field(default_factory=list)
    dictionary: Dict[str, List[str]] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    
    auth: Optional[Auth] = None

    def normalized_goals(self) -> List[Goal]:
        if self.goals:
            return self.goals
        return [self.goal] if self.goal is not None else []



# ========================
# Goal evaluation utilities
# ========================
class GoalStatus(str, Enum):
    MET = "met"
    UNMET = "unmet"


class GoalResult(BaseModel):
    status: GoalStatus
    score: float = 0.0
    reason: Optional[str] = None


class GoalEvaluatorMixin:
    """
    Mixin for a class that has:
      - self.page (Playwright Page)
      - self.model (LLM id for optional screenshot checks)
      - generate_model / generate_text (optional; only used if you enable LLM screenshot check)
    """

    # ---- Deterministic page/heading/URL stop check -----------------
    def _goal_stop_check(self, sw: StopWhen) -> bool:
        try:
            info = self.page.evaluate("""
            () => {
            const u = new URL(location.href);
            const txt = (el) => (el && el.textContent || '').trim().toLowerCase();
            const heads = Array.from(document.querySelectorAll('h1, h2')).map(e => txt(e)).filter(Boolean);
            const title = (document.title || '').toLowerCase();
            const bodyText = (document.body && document.body.innerText || '').replace(/\\s+/g,' ').trim().toLowerCase();
            return { url: u.href.toLowerCase(), hostname: u.hostname.toLowerCase(), path: u.pathname.toLowerCase(),
                    headings: heads, title, bodyText };
            }""")
        except Exception:
            return False

        url   = info.get("url") or ""
        host  = info.get("hostname") or ""
        path  = info.get("path") or ""
        heads = list(info.get("headings") or [])
        title = info.get("title") or ""
        text  = (title + " || " + " ".join(heads) + " || " + (info.get("bodyText") or "")).lower()

        # Per-field matches
        m_url      = any(s.lower() in url   for s in sw.url_contains)   if getattr(sw, "url_contains", None) else None
        m_title    = any(s.lower() in title for s in sw.title_contains) if getattr(sw, "title_contains", None) else None
        m_headings = any(needle.lower() in h for needle in sw.headings for h in ([title] + heads)) if getattr(sw, "headings", None) else None
        m_body     = any(s.lower() in text  for s in sw.body_contains)  if getattr(sw, "body_contains", None) else None
        m_host     = any(s.lower() in host  for s in sw.hostname_in)    if getattr(sw, "hostname_in", None) else None
        m_path     = any(s.lower() in path  for s in sw.path_contains)  if getattr(sw, "path_contains", None) else None

        # Assemble booleans for match_mode
        checks = [bool(m) for m in (m_url, m_title, m_headings, m_body, m_host, m_path) if m is not None]

        # Strong signals gate (if any specified)
        strong_matches = [m for m in (m_url, m_title, m_headings, m_host, m_path) if m is not None]
        if strong_matches:
            if sum(bool(x) for x in strong_matches) < max(0, int(getattr(sw, "min_strong_signals", 0))):
                return False

        # Enforce match_mode over all specified checks (strong+weak)
        primary_ok = (all(checks) if getattr(sw, "match_mode", "any") == "all" else any(checks)) if checks else False
        if not primary_ok:
            return False

        # Optional: require a real URL change
        if getattr(sw, "require_url_change", False):
            try:
                start_url = getattr(self, "_goal_start_url", None)
                if start_url and (start_url.split("#",1)[0] == url.split("#",1)[0]):
                    return False
            except Exception:
                pass

        return True

    
    # ---- Completion signals for submission-like goals --------------
    def _collect_completion_signals(self) -> Dict[str, Any]:
        return self.page.evaluate("""
        () => {
          const txt = (el) => (el && el.textContent || '').trim();
          const title = document.title || '';
          const headings = Array.from(document.querySelectorAll('h1, h2, h3')).map(e => txt(e)).filter(Boolean);
          const bodyText = (document.body && document.body.innerText || '').replace(/\\s+/g,' ').trim();
          const url = location.href;

          const visible = el => {
            const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
            return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none';
          };

          const hasSubmit = Array.from(document.querySelectorAll('button, input[type="submit"], a, [role="button"]'))
            .some(b => visible(b) && /\\b(submit|finish|confirm|apply)\\b/i.test(b.textContent||b.value||''));

          const formsVisible = Array.from(document.querySelectorAll('form')).some(visible);

          const inputs = Array.from(document.querySelectorAll('input, select, textarea'));
          const requiredEmpty = inputs.filter(el => {
            const req = el.required || el.getAttribute('aria-required')==='true' || /\\brequired\\b/i.test(el.outerHTML);
            if (!req || !visible(el) || el.disabled || el.readOnly) return false;
            if (el.matches('input[type=checkbox],input[type=radio]')) {
              const name = el.name;
              const group = name? Array.from(document.querySelectorAll(`input[name="${CSS.escape(name)}"]`)) : [el];
              return !group.some(g => g.checked);
            }
            if (el.tagName==='SELECT') return !el.value;
            return !(el.value && String(el.value).trim().length>0);
          }).length;

          return { url, title, headings, bodyText, hasSubmit, formsVisible, requiredEmpty };
        }
        """)

    def _score_completion(self, signals: Dict[str, Any], policy: CompletionPolicy, screenshot: Optional[bytes]) -> float:
        text = f"{signals.get('title','')} || {' '.join(signals.get('headings',[]))} || {signals.get('bodyText','')}".lower()
        url  = (signals.get('url') or '').lower()
        score = 0.0

        if any(p.lower() in text for p in policy.success_phrases): score += 0.6
        if any(s in url for s in policy.url_success_substrings):  score += 0.2
        if policy.require_no_submit_button and not signals.get('hasSubmit', False): score += 0.1
        if policy.require_form_not_present and not signals.get('formsVisible', True): score += 0.1
        if policy.require_no_required_fields_remaining and int(signals.get('requiredEmpty', 0)) == 0: score += 0.1
        if any(n.lower() in text for n in policy.negative_phrases): score = 0.0

        # Optional LLM screenshot “does this look like a confirmation screen?”
        if policy.use_screenshot_llm_check and screenshot is not None:
            try:
                verdict = self.generate_text(
                    system_prompt=(
                        "You are a strict classifier. Answer strictly 'YES' or 'NO'.\n"
                        "Question: Does this viewport clearly show a successful job application submission/confirmation?"
                    ),
                    prompt="Answer 'YES' or 'NO' only.",
                    model="gpt-5-nano",
                    reasoning_level="low",
                    image_detail="low",
                    image=screenshot
                )
                if str(verdict).strip().upper().startswith("YES"):
                    score += 0.2
            except Exception:
                pass

        return min(score, 1.0)

    # ---- Individual goal evaluators --------------------------------
    def evaluate_page_reached(self, g: PageReachedGoal) -> GoalResult:
        met = self._goal_stop_check(g.stop_when)
        return GoalResult(status=GoalStatus.MET if met else GoalStatus.UNMET,
                          score=1.0 if met else 0.0,
                          reason="StopWhen matched" if met else "StopWhen not matched")

    def evaluate_submission_confirmed(self, g: SubmissionConfirmedGoal) -> GoalResult:
        policy = g.policy or CompletionPolicy()
        sig = self._collect_completion_signals()
        shot = None
        if policy.use_screenshot_llm_check:
            try:
                shot = self.page.screenshot(full_page=False)
            except Exception:
                shot = None
        s = self._score_completion(sig, policy, shot)
        return GoalResult(status=GoalStatus.MET if s >= policy.min_score else GoalStatus.UNMET,
                          score=s, reason=f"score={s:.2f}")
    # --- Robust form completion evaluator (handles iframes & custom widgets) ---

    ATS_IFRAME_HINTS = ("greenhouse.io", "boards.greenhouse.io", "lever.co", "myworkdayjobs.com",
                        "workday.com", "smartrecruiters.com", "icims.com", "jobvite.com",
                        "ashbyhq.com", "bamboohr.com", "oraclecloud.com", "successfactors.com")

    def _form_status_js():
        return FormUtils.form_status_js()

    def _collect_form_status_across_contexts(self):
        """Inspect top page and all *same-origin* iframes. Flag visible cross-origin ATS iframes."""
        # Top document
        top = self.page.evaluate(self._form_status_js())

        required = int(top.get("requiredEmpty") or 0)
        blockers = list(top.get("blockers") or [])
        forms_visible = bool(top.get("formsVisible"))
        has_errors = bool(top.get("hasErrors"))
        has_disabled_submit = bool(top.get("hasDisabledSubmit"))

        # Detect visible cross-origin ATS iframes (we can't inspect inside them)
        cross_origin_ats = []
        for f in (top.get("iframes") or []):
            if not f.get("visible"):
                continue
            src = (f.get("src") or "").lower()
            same = bool(f.get("sameOrigin"))
            if (not same) and any(domain in src for domain in self.ATS_IFRAME_HINTS):
                cross_origin_ats.append(src)

        # Walk same-origin frames and aggregate
        for frame in self.page.frames:
            # Skip the main frame; Playwright's main frame URL equals page.url
            if frame == self.page.main_frame:
                continue
            try:
                data = frame.evaluate(self._form_status_js())
            except Exception:
                # Cross-origin (can’t eval)
                continue
            required += int(data.get("requiredEmpty") or 0)
            blockers += list(data.get("blockers") or [])
            forms_visible = forms_visible or bool(data.get("formsVisible"))
            has_errors = has_errors or bool(data.get("hasErrors"))
            has_disabled_submit = has_disabled_submit or bool(data.get("hasDisabledSubmit"))

        return {
            "requiredEmpty": required,
            "blockers": blockers[:8],
            "formsVisible": forms_visible,
            "hasErrors": has_errors,
            "hasDisabledSubmit": has_disabled_submit,
            "crossOriginATS": cross_origin_ats,
        }

    def evaluate_form_completed(self, g) -> "GoalResult":
        """
        Robust completion test:
        - Counts *real* required-but-empty controls across top document and same-origin iframes.
        - If a visible cross-origin ATS iframe is present (e.g., Greenhouse), we *assume incomplete*
        until we switch into that frame and fill it.
        """
        try:
            status = self._collect_form_status_across_contexts(self)
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"form status error: {e}")

        # Cross-origin ATS iframe visible ⇒ treat as incomplete (can’t verify inside)
        if status["crossOriginATS"]:
            first = status["crossOriginATS"][0]
            return GoalResult(
                status=GoalStatus.UNMET,
                score=0.0,
                reason=f"application form likely inside cross-origin iframe ({first}); switch to that frame and fill it"
            )

        remaining = int(status["requiredEmpty"])
        if remaining > 0:
            # Score decreases as more blockers remain
            score = max(0.0, 1.0 - min(1.0, remaining / 6.0))
            sample = ", ".join(f"{b['label']} ({b['why']})" for b in status["blockers"])
            return GoalResult(
                status=GoalStatus.UNMET,
                score=score,
                reason=f"{remaining} required fields/groups still empty: {sample}"
            )

        # No required blockers; light sanity checks
        if status["hasErrors"]:
            return GoalResult(status=GoalStatus.UNMET, score=0.4, reason="error indicators present on form")
        if status["formsVisible"] and status["hasDisabledSubmit"]:
            return GoalResult(status=GoalStatus.UNMET, score=0.6, reason="submit appears disabled")

        return GoalResult(status=GoalStatus.MET, score=1.0, reason="no required blockers across document/iframes")

    def evaluate_login_completed(self, g: LoginCompletedGoal) -> GoalResult:
        info = self.page.evaluate("""
        () => {
          const visible = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
            return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none'; };
          const text = el => (el && el.textContent || '').trim().toLowerCase();
          const body = (document.body && document.body.innerText || '').toLowerCase();
          const hasLoginForm = Array.from(document.querySelectorAll('input[type="password"], form [type="email"], form [name*="password"]')).some(visible);
          const buttons = Array.from(document.querySelectorAll('a,button,[role="button"]')).filter(visible).map(text);
          return { body, buttons, hasLoginForm };
        }
        """)
        logout_hit = any(any(kw in (b or "") for kw in g.logout_phrases) for b in info["buttons"]) or \
                     any(kw in info["body"] for kw in g.logout_phrases)
        login_absent = (not info["hasLoginForm"]) if g.require_no_login_form else True
        met = logout_hit and login_absent
        return GoalResult(status=GoalStatus.MET if met else GoalStatus.UNMET,
                          score=1.0 if met else 0.0,
                          reason=f"logout_hit={logout_hit}, login_form={info['hasLoginForm']}")

    # ---- Dispatcher & composition ----------------------------------
    def evaluate_goal(self, goal: Goal) -> GoalResult:
        t = goal.type
        if t == GoalKind.PAGE_REACHED:
            return self.evaluate_page_reached(goal)          # type: ignore
        if t == GoalKind.SUBMISSION_CONFIRMED:
            return self.evaluate_submission_confirmed(goal)  # type: ignore
        if t == GoalKind.FORM_COMPLETED:
            return self.evaluate_form_completed(goal)        # type: ignore
        if t == GoalKind.LOGIN_COMPLETED:
            return self.evaluate_login_completed(goal)       # type: ignore
        if t == GoalKind.ANY_OF:
            best = GoalResult(status=GoalStatus.UNMET, score=0.0)
            for g in goal.goals:
                r = self.evaluate_goal(g)
                if r.status == GoalStatus.MET:
                    return r
                if r.score > best.score:
                    best = r
            return best
        if t == GoalKind.ALL_OF:
            scores: List[float] = []
            for g in goal.goals:
                r = self.evaluate_goal(g)
                if r.status != GoalStatus.MET:
                    return r
                scores.append(r.score)
            return GoalResult(status=GoalStatus.MET, score=min(scores) if scores else 1.0)
        return GoalResult(status=GoalStatus.UNMET, score=0.0)

    def goals_met(self, goals: List[Goal], mode: str = "all") -> bool:
        """Convenience: treat top-level as ALL_OF by default."""
        if not goals:
            return False
        if mode == "any":
            return any(self.evaluate_goal(g).status == GoalStatus.MET for g in goals)
        return all(self.evaluate_goal(g).status == GoalStatus.MET for g in goals)


# ---------------------------------------------------------------------------
# Data classes and utilities
# ---------------------------------------------------------------------------
    
@dataclass
class ViewportMeta:
    width: int
    height: int
    dpr: float
    scroll_x: int
    scroll_y: int
    ss_pixel_w: int
    ss_pixel_h: int
    css_scale: float
    doc_width: int
    doc_height: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "viewport": {"width": self.width, "height": self.height, "dpr": self.dpr, "scrollX": self.scroll_x, "scrollY": self.scroll_y, "docWidth": self.doc_width, "docHeight": self.doc_height},
            "screenshot": {"pixelWidth": self.ss_pixel_w, "pixelHeight": self.ss_pixel_h, "cssScale": self.css_scale},
            "grid": {"major": 50, "minor": 10},
        }

class ExpandedPrompt(BaseModel):
    expanded_prompt: str
    assumptions: List[str]
    validation_checklist: List[str]

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _denormalize_from_1000(coord: int, dimension: int) -> int:
    """Convert normalized 0-1000 coordinate back to pixels"""
    return int((coord / 1000.0) * dimension)

def _get_item_center_pixels(item: DetectedItem, width: int, height: int) -> Tuple[int, int]:
    """Get the center coordinates of a detected item in pixel coordinates"""
    ymin, xmin, ymax, xmax = item.box_2d
    
    # Convert normalized coordinates to pixels
    pixel_xmin = _denormalize_from_1000(xmin, width)
    pixel_ymin = _denormalize_from_1000(ymin, height)
    pixel_xmax = _denormalize_from_1000(xmax, width)
    pixel_ymax = _denormalize_from_1000(ymax, height)
    
    # Calculate center
    center_x = (pixel_xmin + pixel_xmax) // 2
    center_y = (pixel_ymin + pixel_ymax) // 2
    
    return center_x, center_y

def _draw_detection_boxes(image_bytes: bytes, detection_result: DetectionResult) -> bytes:
    """Draw bounding boxes around detected items on the image"""
    im = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(im)
    
    width, height = im.size
    
    for i, item in enumerate(detection_result.items):
        ymin, xmin, ymax, xmax = item.box_2d
        
        # Convert normalized coordinates to pixels
        pixel_xmin = _denormalize_from_1000(xmin, width)
        pixel_ymin = _denormalize_from_1000(ymin, height)
        pixel_xmax = _denormalize_from_1000(xmax, width)
        pixel_ymax = _denormalize_from_1000(ymax, height)
        
        # Choose color based on item type
        color = (255, 0, 0) if item.clickable else (0, 255, 0)  # Red for clickable, green for others
        
        # Draw bounding box
        draw.rectangle([pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax], 
                      outline=color, width=2)
        
        # Draw index number
        draw.text((pixel_xmin + 2, pixel_ymin + 2), str(i), fill=color, font=None)
        
        # Draw confidence
        conf_text = f"{item.confidence:.2f}"
        draw.text((pixel_xmin + 2, pixel_ymax - 15), conf_text, fill=color, font=None)
    
    out = io.BytesIO()
    im.save(out, format="PNG")
    
    return_value = out.getvalue()
    with open("screenshot_with_detections.png", "wb") as f:
        f.write(return_value)
    return return_value

class VisionCoordinator(GoalEvaluatorMixin):
    """Coordinate-only Playwright driver using a structured-output vision model.

    Parameters
    ----------
    page : Playwright Page
        The Playwright page to control.
    model : str
        Model name to pass to your GPT function (default: "gpt-5-nano").
    reasoning_level : str
        e.g., "low" | "medium" | "high" (passed through to your GPT function).
    use_grid : bool
        If True, overlays a faint grid on the screenshot sent to the model.
    viewport_lock : bool
        If True, verifies DPR==1 and raises if not (you can relax this if needed).
    image_detail : str
        Vision detail hint (e.g., "low" | "high").
    """

    def __init__(
        self,
        page: Page,
        model: str = "gpt-5-nano",
        reasoning_level: str = "low",
        viewport_lock: bool = True,
        image_detail: str = "high",
        preferences: dict = None,
    ) -> None:
        self.page = page
        self.model = model
        self.reasoning_level = reasoning_level
        self.viewport_lock = viewport_lock
        self.image_detail = image_detail
        self.preferences = preferences
        
    # ---------- tiny utils ----------

    def _dedupe(self, seq: List[str]) -> List[str]:
        return TextUtils.dedupe(seq)

    def _extract_urls(self, text: str) -> List[str]:
        return TextUtils.extract_urls(text)

    def _to_num(self, tok: str) -> Optional[int]:
        return TextUtils.to_num(tok)

    def _slugify(self, bits: List[str]) -> str:
        return TextUtils.slugify(bits)

    def _looks_like_navigation(self, low: str) -> bool:
        return TextUtils.looks_like_navigation(low)

    def _looks_like_form_fill(self, low: str) -> bool:
        return TextUtils.looks_like_form_fill(low)

    def _looks_like_login(self, low: str) -> bool:
        return TextUtils.looks_like_login(low)

    def _looks_like_submit(self, low: str) -> bool:
        return TextUtils.looks_like_submit(low)

    def _guess_cta_preferences(self, low: str) -> List[str]:
        prefs = TextUtils.guess_cta_preferences(low)
        return self._dedupe(prefs)

    # ---------- StopWhen inference ----------
    DOMAIN_RX = re.compile(r"\b([a-z0-9.-]+\.[a-z]{2,})(?:/[\w\-./?%&=]*)?\b", re.I)

    def _is_navigation_intent(self, text: str) -> bool:
        t = text.lower().strip()
        NAV_VERBS = ("go to", "goto", "navigate to", "open", "visit", "take me to", "go ", "navigate ", "open ")
        # loose, but fast
        return any(t.startswith(v) for v in NAV_VERBS) or "page" in t or "site" in t or "website" in t

    def _guess_stopwhen_for_navigation(self, subprompt: str) -> StopWhen:
        sp = subprompt.lower()

        # Try to extract a domain if present
        m = self.DOMAIN_RX.search(sp)
        hostname = m.group(1).lower() if m else None

        # Heuristics for "about" / "careers" / etc.
        want_about = any(k in sp for k in ("about", "about us"))
        path_hints = []
        if want_about:
            path_hints.append("/about")

        if hostname:
            return StopWhen(
                hostname_in=[hostname],
                path_contains=path_hints,
                title_contains=["about"] if want_about else [],
                # Require actual nav + at least one strong signal
                require_url_change=True,
                min_strong_signals=1,
                match_mode="any",
            )
        # No domain? Fall back to strong signals only; avoid body_contains
        return StopWhen(
            title_contains=["about"] if want_about else [],
            headings=["about"] if want_about else [],
            min_strong_signals=1,
            match_mode="any",
        )
        
    def _guess_stop_when(self, prompt: str) -> StopWhen:
        """
        Heuristics: pull obvious nouns+numbers (“sprint 5”, “step 2”, “section three”), quoted titles, and url slugs.
        Produces a non-empty StopWhen for common nav prompts like “navigate to sprint 5 page”.
        """
        low = (prompt or "").lower()
        headings: List[str] = []
        title_contains: List[str] = []
        body_contains: List[str] = []
        url_contains: List[str] = []

        # quoted things → strong heading/title cues
        for m in re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,80})[\"“”'‘’]", prompt or ""):
            t = m.strip()
            if t:
                headings.append(t)
                title_contains.append(t)

        # nouns with ordinals: sprint/step/section/chapter/page
        for m in re.finditer(r"\b(sprint|step|section|chapter|page)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", low):
            noun, numtok = m.group(1), m.group(2)
            num = self._to_num(numtok)
            if not num: continue
            pretty = f"{noun.title()} {num}"
            headings.append(pretty)
            title_contains.append(pretty)
            url_contains.extend([f"{noun}-{num}", f"{noun}{num}"])

        # generic “dashboard/confirmation/thank you”
        if any(k in low for k in ["dashboard"]):
            headings.append("Dashboard"); title_contains.append("Dashboard")
        if any(k in low for k in ["thank you", "confirmation"]):
            body_contains.append("thank you"); title_contains.append("confirmation")

        # URLs mentioned directly
        for u in self._extract_urls(prompt or ""):
            try:
                p = self.urlparse(u)
                if p.netloc: url_contains.append(p.netloc.lower())
                if p.path:   url_contains.append(self._slugify(p.path.split("/")))
            except Exception:
                pass

        return StopWhen(
            headings=self._dedupe(headings),
            url_contains=self._dedupe([u for u in url_contains if u]),
            title_contains=self._dedupe(title_contains),
            body_contains=self._dedupe(body_contains),
            aria_active_tab=[],
            zero_required_fields_remaining=False,
            match_mode="any",
        )

    def _ai_stopwhen_from_prompt(self, prompt: str) -> Optional[StopWhen]:
        """
        Optional “last resort” AI inference for StopWhen when heuristics find nothing useful.
        Keeps it minimal: headings/title/url substrings only.
        """
        try:
            sys = (
                "Infer minimal robust StopWhen fields for the user goal.\n"
                "Only set: headings, url_contains, title_contains, body_contains; leave others empty.\n"
                "Prefer nouns/labels likely visible on the page (e.g., 'Sprint 5', 'Dashboard') and url slugs (e.g., 'sprint-5')."
            )
            sp = generate_model(
                prompt=f"Goal: {prompt}\nReturn a strictly valid JSON object for StopWhen.",
                model_object_type=StopWhen,
                system_prompt=sys,
                model="gpt-5-nano",
                reasoning_level="low",
            )
            return sp
        except Exception:
            return None

    # ---------- DOM → Adapter synthesizer ----------
    def _synthesize_adapter_from_dom(self, vc: "VisionCoordinator", name: Optional[str] = None) -> Optional[Adapter]:
        """
        Builds a lightweight Adapter from visible DOM. Safe if cross-origin blocks occur.
        """
        try:
            host = ""
            try:
                host = self.urlparse(vc.page.url).netloc
            except Exception:
                pass
            data = vc.page.evaluate("""
            () => {
            const visible = el => {
                if (!el) return false;
                const r = el.getBoundingClientRect();
                const s = getComputedStyle(el);
                return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none';
            };
            const texts = el => (el.textContent||el.value||'').trim();

            const buttons = Array.from(document.querySelectorAll('button, [role="button"], input[type="submit"], a'))
                .filter(visible)
                .map(el => texts(el))
                .filter(Boolean)
                .map(t => t.replace(/\\s+/g,' ').trim())
                .slice(0, 150);

            const requiredMarkers = ["*", "required", "aria-required=true"];
            const uploadLabels = Array.from(document.querySelectorAll('label, [aria-label], [for], [title]'))
                .filter(visible)
                .map(el => (el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '').trim())
                .filter(Boolean)
                .filter(t => /resume|cv|cover\\s*letter|upload|attachment/i.test(t))
                .slice(0, 50);

            const stepper = Array.from(document.querySelectorAll('nav, [role="tablist"], [class*="step"], [class*="progress"]'))
                .filter(visible)
                .flatMap(el => Array.from(el.querySelectorAll('*')).map(n => (n.textContent||'').trim()))
                .filter(Boolean)
                .filter(t => /sprint\s*\d+|step\s*\d+|stage\s*\d+|page\s*\d+/i.test(t))
                .slice(0, 20);

            return { buttons, uploadLabels, requiredMarkers, stepper };
            }
            """)

            btns = data.get("buttons", [])
            progress = [t for t in btns if re.search(r"\b(next|continue|back|previous|apply|apply now|proceed)\b", t, re.I)]
            submits  = [t for t in btns if re.search(r"\b(submit|send|finish|confirm|place order|checkout|pay now|complete)\b", t, re.I)]

            hints = AdapterHints(
                progress_buttons=self._dedupe(progress)[:20],
                submit_buttons=self._dedupe(submits)[:20],
                required_markers=self._dedupe(data.get("requiredMarkers", []))[:10],
                stepper_labels=self._dedupe(data.get("stepper", []))[:12],
                upload_labels=self._dedupe(data.get("uploadLabels", []))[:12],
            )
            detect = AdapterDetect(url_contains=[host] if host else [])
            return Adapter(name=name or (host or "AutoAdapter"), detect=detect, hints=hints)
        except Exception:
            return None

    # ---------- default profile + dictionary ----------
    def _synthesize_profile_and_dictionary(
        self,
        prompt: str,
        base_profile: Optional[dict],
        base_dictionary: Optional[dict],
    ) -> Tuple[dict, dict]:
        """
        Neutral defaults; if caller passed values, we just merge.
        """
        default_profile = {
            "first_name": "Alex",
            "last_name":  "Doe",
            "email":      "alex@example.com",
            "phone":      "+10000000000",
            "location":   "",
            "linkedin":   "",
            "github":     "",
            "website":    "",
        }
        synonyms = {
            "first_name": ["first name", "given name"],
            "last_name":  ["last name", "surname", "family name"],
            "email":      ["email", "e-mail"],
            "phone":      ["phone", "mobile", "telephone"],
            "location":   ["location", "city", "current location", "where you live"],
            "linkedin":   ["linkedin", "linkedin profile"],
            "github":     ["github"],
            "website":    ["website", "portfolio", "personal site"],
        }

        prof = dict(default_profile)
        if base_profile: prof.update(base_profile)

        dct = {k: list(v) for k, v in synonyms.items()}
        for k, v in (base_dictionary or {}).items():
            # merge while deduping
            dct[k] = self._dedupe((dct.get(k, []) + list(v)))

        # if prompt looks login-ish, prefer username/password placeholders
        low = (prompt or "").lower()
        if self._looks_like_login(low):
            prof.setdefault("username", "user@example.com")
            prof.setdefault("password", "hunter2")

        return prof, dct

    # ---------- main: infer_hints_for_subprompt ----------
    def infer_hints_for_subprompt(
        self,
        subprompt: str,
        *,
        profile: dict | None = None,
        dictionary: dict | None = None,
        adapters: List[Adapter] | None = None,
        assets: Assets | None = None,
        base_policies: Policies | None = None,
        use_ai_stopwhen_fallback: bool = True,
    ) -> UXHints:
        s = (subprompt or "").strip()
        low = s.lower()

        # intents
        wants_nav    = self._looks_like_navigation(low) or bool(self._extract_urls(s))
        wants_form   = self._looks_like_form_fill(low)
        wants_login  = self._looks_like_login(low)
        wants_submit = self._looks_like_submit(low)  # explicit only

        # StopWhen
        sw = self._guess_stop_when(s)
        if use_ai_stopwhen_fallback and not (sw.headings or sw.url_contains or sw.title_contains or sw.body_contains):
            ai_sw = self._ai_stopwhen_from_prompt(s)
            if ai_sw: sw = ai_sw

        # goals
        goals: List[Goal] = []
        if wants_nav:   goals.append(PageReachedGoal(stop_when=sw))
        if wants_form:  goals.append(FormCompletedGoal())
        if wants_login: goals.append(LoginCompletedGoal())
        if wants_submit:
            goals.append(SubmissionConfirmedGoal(policy=CompletionPolicy(intent=s[:128])))

        if not goals:
            goals = [PageReachedGoal(stop_when=sw)]
        elif len(goals) > 1:
            goals = [AllOfGoal(goals=goals)] if " or " not in low else [AnyOfGoal(goals=goals)]

        # progress policy
        prefer = self._guess_cta_preferences(low)
        avoid: List[str] = []

        # adapters: use provided, or synthesize from DOM
        adapters_in: List[Adapter] = list(adapters or [])
        if not adapters_in:
            auto_ad = self._synthesize_adapter_from_dom(self)
            if auto_ad: adapters_in.append(auto_ad)

        for ad in adapters_in:
            if ad and ad.hints:
                prefer.extend(ad.hints.progress_buttons or [])
                avoid.extend(ad.hints.submit_buttons or [])

        fill_scope = "all" if any(k in low for k in ["all fields", "entire form", "everything", "complete all"]) else "required"
        progress = ProgressPolicy(
            fill_only=fill_scope,
            prefer_cta_text=self._dedupe(prefer),
            avoid_cta_text=self._dedupe(avoid),
            login_allowed=True,
        )

        # policies
        if base_policies is not None:
            policies = Policies(**base_policies.model_dump())
        else:
            policies = Policies(
                allow_submit=False,
                captcha="stop",
                destructive_actions="block",
                privacy=PrivacyPolicy(pii_budget="minimal", redact_logs=True),
                synthesis=SynthesisPolicy(strategy="safe_defaults", style="neutral", max_chars=200),
                completion=None,
            )
        if wants_submit:
            policies.allow_submit = True

        # assets
        if assets is None:
            if any(k in low for k in ["upload", "attach", "choose file", "select file", "attachment", "resume", "cv"]):
                assets = Assets(
                    resume_path=None,
                    cover_letter_path=None,
                    upload_policy=UploadPolicy(
                        allow_pdf=True, allow_docx=True, allow_images=True,
                        max_mb=50,
                        prefer=["resume_path", "cover_letter_path"],
                        match_by=["upload", "file", "choose file", "select file", "attachment", "resume", "cv"],
                        if_multiple_uploads_required="first",
                    ),
                )
            else:
                assets = Assets(upload_policy=UploadPolicy())

        # profile + dictionary (merge caller-provided with defaults)
        prof, dct = self._synthesize_profile_and_dictionary(subprompt, profile, dictionary)

        # constraints (ban submits unless explicitly allowed)
        constraints: List[str] = []
        if not policies.allow_submit:
            submit_words = [
                "Submit", "Send", "Finish", "Confirm",
                "Place order", "Pay now", "Checkout", "Complete purchase",
            ]
            for ad in adapters_in:
                if ad and ad.hints and ad.hints.submit_buttons:
                    submit_words.extend(ad.hints.submit_buttons)
            submit_words = self._dedupe(submit_words)
            def _quote(s: str) -> str:
                return re.sub(r"([.^$*+?{}\\[\\]|()])", r"\\\1", s)
            pattern = "|".join(_quote(w) for w in submit_words)
            constraints.append(f"Never click buttons matching: {pattern}")

        if self._is_navigation_intent(s):
                sw = self._guess_stopwhen_for_navigation(s)
                return UXHints(
                    goals=[PageReachedGoal(stop_when=sw)],
                    goal_progress=ProgressPolicy(  # safe defaults for nav
                        fill_only="none",
                        prefer_cta_text=["Continue", "Next"],
                        login_allowed=False,
                    ),
                    policies=policies,
                    profile={},
                    dictionary={},
                    adapters=[],
                    assets=Assets(),
                    constraints=[
                        "Never click buttons matching: Submit|Send|Finish|Confirm|Place order|Pay now|Checkout|Complete purchase"
                    ],
                )

        return UXHints(
            goals=goals,
            goal_progress=progress,
            policies=policies,
            profile=prof,
            dictionary=dct,
            assets=assets,
            adapters=adapters_in,
            constraints=constraints,
        )

    def parse_task_into_subtasks(self, task_hint: str) -> TaskDecomposition:
            """
            Break down the task into a list of sub-tasks.
            Splits on common sequencing words (then, next, after that, finally, etc.)
            in a case-insensitive and punctuation-tolerant way.
            """
            system_prompt = f"""
                You are a task decomposition engine. 
                Your job is to break down the provided task into a list of clear, atomic sub-tasks.

                RULES
                1. Split the task into smaller steps whenever connected by "then", commas, "and", or other conjunctions.
                2. Each sub-task must be a standalone instruction, written in natural language.
                3. Preserve the original intent — do not drop, merge, or rephrase tasks in a way that changes their meaning.
                4. If the task is already atomic (single clear instruction), return it unchanged in a one-element list.
                5. If the task contains general actions like "click all the dropdowns" or "find all the inputs related to a person's name", keep them as a single item (do not expand unless explicitly listed).
                6. Always return the final result as a strict JSON array of strings.

                INPUT TASK
                {task_hint}

                EXAMPLES
                Task: "click the accept all cookies button then click the apply now button then type hello in the first name field then type world in the last name field"
                Output: ["click the accept all cookies button", "click the apply now button", "type hello in the first name field", "type world in the last name field"]

                Task: "click the accept all cookies button, click the apply now button"
                Output: ["click the accept all cookies button", "click the apply now button"]

                Task: "find and fill all the inputs related to a person's name, accept any terms then click the apply now button"
                Output: ["find all the inputs related to a person's name", "accept any terms", "click the apply now button"]

                Task: "click all the dropdowns"
                Output: ["click all the dropdowns"]

                Task: "click the apply now button"
                Output: ["click the apply now button"]

                OBJECTIVE
                Return ONLY the JSON array of sub-tasks. Do not include any other text.
                """
            
            user_prompt = """
            Return the list of sub-tasks. Do not include any other text.
            """
            if not task_hint or not task_hint.strip():
                return TaskDecomposition(task=task_hint, sub_tasks=[])

            result = generate_model(
                prompt=user_prompt,
                model_object_type=TaskDecomposition,
                system_prompt=system_prompt,
                model="gpt-5-mini",
                reasoning_level="low",
            )
            
            return result
    
    def _merge_preferences(self, base: Dict[str, Any], hints: UXHints) -> Dict[str, Any]:
        base = dict(base or {})
        # keep structured so the planner can see it
        base.setdefault("profile", {})
        base["profile"].update(hints.profile or {})
        base.setdefault("assets", {})
        base["assets"].update(hints.assets.model_dump())
        if hints.auth:
            base["auth"] = hints.auth.model_dump()
        if hints.dictionary:
            base["dictionary"] = hints.dictionary
        # expose policies to the planner (for synthesis etc.)
        base["policies"] = hints.policies.model_dump()
        return base


    def _active_adapter_for_url(self, url: str, hints: UXHints) -> Optional[Adapter]:
        for ad in hints.adapters or []:
            if any(sub in url for sub in ad.detect.url_contains):
                return ad
        return None


    def _humanize_stop_conditions(self, sw: StopWhen) -> str:
        parts = []
        if sw.zero_required_fields_remaining:
            parts.append("all required form fields are filled")
        if sw.headings:
            parts.append("heading shows any of: " + ", ".join(f'"{h}"' for h in sw.headings))
        if sw.url_contains:
            parts.append("URL contains any of: " + ", ".join(f'"{u}"' for u in sw.url_contains))
        if sw.title_contains:
            parts.append("title contains any of: " + ", ".join(f'"{t}"' for t in sw.title_contains))
        if sw.hostname_in:
            parts.append("hostname is any of: " + ", ".join(f'"{h}"' for h in sw.hostname_in))
        if sw.path_contains:
            parts.append("path contains any of: " + ", ".join(f'"{p}"' for p in sw.path_contains))
        if sw.body_contains:
            parts.append("body contains (weak): " + ", ".join(f'"{b}"' for b in sw.body_contains))
        if sw.require_url_change:
            parts.append("URL changed from start")
        if sw.min_strong_signals > 1:
            parts.append(f"≥{sw.min_strong_signals} strong signals")
        return "; ".join(parts) or "—"



    def _snapshot_from_hints(self, h: UXHints, active: Optional[Adapter]) -> str:
        gp = h.goal_progress or ProgressPolicy()
        stop = "—"
        first_goal = h.goals[0] if h.goals else None
        if first_goal and hasattr(first_goal, "stop_when"):
            stop = self._humanize_stop_conditions(first_goal.stop_when)

        up = h.assets.upload_policy
        upload_summary = f"prefer: {', '.join(up.prefer) or '—'}; match_by: {', '.join(up.match_by) or '—'}; multi: {up.if_multiple_uploads_required}"
        dict_keys = ", ".join(f"{k}:{'|'.join(v)}" for k, v in (h.dictionary or {}).items()) or "—"
        synth = h.policies.synthesis
        synth_summary = f"{synth.strategy} (style={synth.style}, max_chars={synth.max_chars})"

        adapter_summary = ""
        if active:
            ah = active.hints
            adapter_summary = (
                f"\n- Adapter: {active.name}\n"
                f"  progress_buttons: {', '.join(ah.progress_buttons) or '—'}\n"
                f"  submit_buttons: {', '.join(ah.submit_buttons) or '—'}\n"
                f"  required_markers: {', '.join(ah.required_markers) or '—'}\n"
                f"  upload_labels: {', '.join(ah.upload_labels) or '—'}\n"
                f"  stepper_labels: {', '.join(ah.stepper_labels) or '—'}"
            )

        prefer = ", ".join(gp.prefer_cta_text) or "—"
        avoid  = ", ".join(gp.avoid_cta_text) or "—"
        fill_scope = gp.fill_only

        return (
            f"- Stop when: {stop}\n"
            f"- Prefer CTAs: {prefer}\n"
            f"- Avoid CTAs: {avoid}\n"
            f"- Fill scope: {fill_scope}\n"
            f"- Upload policy: {upload_summary}\n"
            f"- Dictionary: {dict_keys}\n"
            f"- Synthesis: {synth_summary}"
            f"{adapter_summary}"
        )


    def _build_hard_constraints(self, meta_instructions: str, hints: UXHints, active: Optional[Adapter]) -> str:
        """Combine user meta_instructions + policy bans + adapter submit buttons."""
        lines = []

        # 1) Only ban-style lines from meta_instructions
        meta_bans = self._extract_ban_constraints(meta_instructions or "")
        if meta_bans:
            lines.append(meta_bans)

        # 2) Policy bans
        if not hints.policies.allow_submit:
            # merge policy avoid list + adapter submit buttons
            first_goal = hints.goals[0] if hints.goals else None
            if first_goal and hasattr(first_goal, 'progress_policy'):
                submit_words = set(first_goal.progress_policy.avoid_cta_text)
                if active and active.hints.submit_buttons:
                    submit_words.update(active.hints.submit_buttons)
                if submit_words:
                    lines.append("Never click final/commit buttons matching: " + "|".join(sorted(submit_words)))

        if hints.policies.destructive_actions == "block":
            lines.append("Never click destructive actions: Delete|Remove|Reset|Discard|Cancel application")

        if hints.policies.captcha == "stop":
            lines.append("If CAPTCHA or bot verification appears, stop and return empty steps.")

        # 3) User constraints (only ban-ish lines will get through anyway if you run them through _extract_ban_constraints)
        for c in hints.constraints or []:
            # optionally run through the same extractor so we don't leak goals into bans
            c_ban = self._extract_ban_constraints(c)
            if c_ban:
                lines.append(c_ban)

        return "\n".join(l for l in lines if l)

    def _extract_ban_constraints(self, meta_instructions: str) -> str:
        """
        Keep only 'ban' style lines for the verifier. Everything else (GOAL/STOP/PROGRESSION)
        should guide planning but must NOT cause violations when taking intermediate steps.
        """
        if not meta_instructions:
            return ""
        bans = []
        for raw in meta_instructions.splitlines():
            line = raw.strip()
            low  = line.lower()
            if not line:
                continue
            if low.startswith(("do not", "don't", "never", "forbid", "ban", "avoid")):
                bans.append(line)
            # Optionally: catch 'Never click' even if not at line start
            elif "never click" in low:
                bans.append(line)
        return "\n".join(bans)
    
    def evaluate_page_reached(self, g: PageReachedGoal) -> GoalResult:
        met = self._goal_stop_check(g.stop_when)
        return GoalResult(
            status=GoalStatus.MET if met else GoalStatus.UNMET,
            score=1.0 if met else 0.0,
            reason="StopWhen matched" if met else "StopWhen not matched",
        )
        
    def _is_required_control_at_point(self, x: int, y: int) -> bool:
        """
        Heuristic: at (x,y) find nearest form control and decide if it's required.
        Considers: @required, aria-required, label with "*" or "required".
        """
        try:
            return bool(self.page.evaluate("""
            (x, y) => {
            const visible = el => {
                if (!el) return false;
                const r = el.getBoundingClientRect();
                const s = getComputedStyle(el);
                return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none';
            };
            const inRange = (v, min, max) => v>=min && v<=max;

            // Probe several points around (x,y) to be robust to small mis-centers
            const probes = [[0,0],[0,-8],[0,8],[-8,0],[8,0],[-12,0],[12,0]];
            for (const [dx,dy] of probes) {
                const els = document.elementsFromPoint(x+dx, y+dy);
                for (const el of els) {
                const ctrl = el.closest && el.closest('input,select,textarea,[role="combobox"],[contenteditable="true"]');
                if (!ctrl || !visible(ctrl)) continue;

                // checkbox/radio groups are handled implicitly: if any is required, label often marks with *
                let req = !!(ctrl.required || ctrl.getAttribute('aria-required')==='true' || ctrl.matches('[data-required="true"]'));

                if (!req) {
                    const id = ctrl.id;
                    if (id) {
                    const lab = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                    if (lab) {
                        const t = (lab.textContent||'').toLowerCase();
                        if (/\\*/.test(t) || /required/.test(t)) req = true;
                    }
                    }
                    // Fallback: look up the DOM for a nearby label
                    if (!req) {
                    const lab2 = ctrl.closest('label');
                    if (lab2) {
                        const t = (lab2.textContent||'').toLowerCase();
                        if (/\*/.test(t) || /required/.test(t)) req = true;
                    }
                    }
                }
                return !!req;
                }
            }
            return false;
            }
            """, x, y))
        except Exception:
            return False

    def _apply_fill_policy(self, plan: ModelPlan, meta: ViewportMeta, fill_only: str) -> ModelPlan:
        """
        If fill_only == 'required', drop fill/select/check steps that target non-required fields.
        """
        if not plan or not plan.steps or fill_only != "required":
            return plan

        kept = []
        items = plan.detection_result.items if plan and plan.detection_result else []
        for s in plan.steps:
            # Only gate field-changing actions. Navigation clicks (Next/Continue) should pass.
            gate = s.action in ("type",) or \
                (s.action == "click" and s.target_item_index is not None and
                    0 <= s.target_item_index < len(items) and
                    (items[s.target_item_index].input_type in ("text","email","password","checkbox","radio","select","upload")))
            if not gate:
                kept.append(s); continue

            # Resolve coordinates for required test
            x, y = None, None
            if s.target_item_index is not None and 0 <= s.target_item_index < len(items):
                cx, cy = _get_item_center_pixels(items[s.target_item_index], meta.width, meta.height)
                x, y = cx, cy
            elif s.x is not None and s.y is not None:
                x, y = int(s.x), int(s.y)

            if x is None or y is None:
                # can't judge → keep (conservative)
                kept.append(s); continue

            is_required = self._is_required_control_at_point(x, y)
            if is_required:
                kept.append(s)
            else:
                print(f"[fill-policy] Dropping step on non-required control at ({x},{y})")
        plan.steps = kept
        return plan

    
    def _exec_compressed_phrase(
    self,
    prompt: str,
    meta_instructions: str = "",
    attempts: int = 30,
    hints: UXHints | Dict[str, Any] | None = None,   # NEW
):
        """
        Executes a compressed phrase using ONE-PASS detection+selection+planning per viewport.
        - NEW: accepts `hints` (UXHints) to specify goals/stop conditions, profile values, assets, policies, adapters, synonyms.
        - Merges hints into HARD CONSTRAINTS and into preferences for value sourcing.
        - Continues to reset state on navigation and enforces constraints with verifier + local filters.
        """
        from pydantic import ValidationError

        # Normalize hints
        if hints is None:
            hints = UXHints()
        elif isinstance(hints, dict):
            try:
                hints = UXHints.model_validate(hints)
            except ValidationError as e:
                print(f"[hints] validation failed, proceeding with defaults: {e}")
                hints = UXHints()

        # ---- Config ----
        ONEPASS_RETRIES = 4
        MAX_ACTIONS = 60
        MAX_REPEAT_PLAN = 3
        MAX_SAME_SHOTS = 3
        VIEWPORT_OVERLAP = 0.75
        HEIGHT_GROWTH_FACTOR = 1.1
        
        goals = hints.normalized_goals()

        # ---- Helpers (subset; reusing your previous robust ones where applicable) ----
        def _active_adapter() -> Optional[Adapter]:
            return self._active_adapter_for_url(self.page.url, hints)

        def _unified_system_prompt(
            target_prompt, meta: ViewportMeta, preferences, do_not_touch, hard_constraints, hint_snapshot
        ):
            active = _active_adapter()
            gp = hints.goal_progress or ProgressPolicy()

            adapter_stepper = ""
            if active and active.hints.stepper_labels:
                adapter_stepper = f"\n- Stepper labels (adapter): {', '.join(active.hints.stepper_labels)}"

            first_goal = hints.goals[0] if hints.goals else None
            stop_when = self._humanize_stop_conditions(first_goal.stop_when) if first_goal and hasattr(first_goal, 'stop_when') else "—"

            return f"""
                    ROLE
                    You are a STRICT, DETERMINISTIC UI Vision+Planner. In ONE PASS on the provided screenshot you must:
                    (A) Detect prominent UI items,
                    (B) Select only items relevant to the user's intent,
                    (C) Output an ordered plan of steps to achieve the intent.

                    PRIORITY OF INSTRUCTIONS (highest → lowest)
                    1) HARD CONSTRAINTS (below)  ← MUST be followed exactly
                    2) target_prompt
                    3) preferences/hints
                    4) general rules below

                    INPUTS
                    - target_prompt: {target_prompt}
                    - preferences: {preferences}
                    - meta: {meta.as_dict()}
                    - do_not_touch (boxes): {do_not_touch or []}

                    HARD CONSTRAINTS (MUST FOLLOW EXACTLY):
                    \"\"\"{hard_constraints}\"\"\"

                    GOAL & STOP CONDITIONS
                    - Consider the task completed when: {stop_when}.
                    - If completed, return empty steps.

                    HINTS SNAPSHOT
                    {hint_snapshot}{adapter_stepper}

                    SCOPE (PROMINENT VIEW)
                    1) If a blocking modal exists, scope to the top-most modal only.
                    2) Otherwise, main document; exclude hidden/disabled/off-viewport/inert/aria-hidden/obscured elements.

                    DETECTION REQUIREMENTS
                    - Detect ALL prominent interactive/visual elements.
                    - Each item MUST include:
                    - description; viewport_data (copy meta)
                    - box_2d [ymin,xmin,ymax,xmax] normalized 0–1000 (ints)
                    - section_in one of ["header","footer","sidebar","content","modal","navigation","other"]
                    - confidence 0.0–1.0
                    - triggers_file_input (bool)
                    - item_type (button, link, input, text, image, icon, checkbox, radio, select, combobox, tab, switch, slider)
                    - input_type ("text","email","password","upload","radio","checkbox","select","not an input")
                    - value "" if no current value
                    - clickable (bool)
                    - surrounding_context
                    - in_dialog (bool)
                    - is_behind_modal (bool)

                    SELECTION (RELEVANCE)
                    - Match via visible labels/placeholder/aria-label/title/value + nearby headings.
                    - Prefer CTAs in hints.goal_progress.prefer_cta_text: {', '.join(gp.prefer_cta_text) or '—'}
                    - EXCLUDE do_not_touch overlaps (IoU>0.25) or identical elements.
                    - Treat adapter submit buttons as high-risk unless explicitly allowed by constraints.

                    PLANNING
                    - Use indices into detection_result.items; x,y fallback allowed.
                    - If off-screen, add a scroll then a wait before interaction.
                    - Filling policy: {gp.fill_only} fields.
                    - VALUE SOURCING ORDER:
                    1) preferences.profile exact field or synonyms (preferences.dictionary)
                    2) If missing and policies.synthesis.strategy == "safe_defaults" or "model_generate", synthesize within max_chars; else leave blank.
                    - Uploads: match inputs using upload_policy.match_by; try assets in upload_policy.prefer order.
                    - Login: allowed={gp.login_allowed}. If login page visible and auth provided, perform necessary login steps safely.
                    - If destructive/ambiguous and not allowed by constraints, STOP (empty steps).

                    QUALITY & SAFETY
                    - Only visible, enabled, unobscured items.
                    - If ANY HARD CONSTRAINT conflicts with an otherwise-valid action, DO NOT ACT. Return empty steps.

                    RETURN FORMAT (STRICT JSON → ModelPlan)
                    - detection_result: DetectionResult
                    - steps: List[Step]
                    - reason: must include "constraints: OK" or "constraints: VIOLATED → <which>"
                    - confidence: 0.0–1.0

                    CRITICAL
                    - Output ONLY valid JSON conforming to the schema. No prose outside JSON.
                    """.strip()

        def _compile_meta_rules(hard_constraints: str, active: Optional[Adapter]):
            """Local filters that never execute forbidden actions (belt & braces).

            If `active` is falsy (None/False), no constraints are applied.
            """
            rules = {"ban_click_text": [], "ban_actions": []}

            # --- Active filter ---
            # If there is no active adapter (or caller passed False/None), skip generating constraints.
            if active in (None, False, 0):
                return rules

            # Parse "Never click ... matching: a|b|c" lines
            for line in (hard_constraints or "").splitlines():
                low = line.lower()
                if low.startswith("never click"):
                    if "matching:" in low:
                        parts = line.split("matching:", 1)[1].strip()
                        tokens = [t.strip() for t in re.split(r"[|]", parts) if t.strip()]
                        rules["ban_click_text"].extend(tokens)

            # Extra: block password typing if policies say so (keep consistent with your earlier code if needed)
            return rules

        def _enforce_rules_locally(plan: "ModelPlan", rules, items: list["DetectedItem"]):
            if not plan or not plan.steps:
                return plan
            def _txt(i):
                try:
                    it = items[i]
                    return (f"{it.description} {it.surrounding_context or ''}").lower()
                except Exception:
                    return ""
            banned = [re.compile(re.escape(s), re.I) for s in (rules.get("ban_click_text") or [])]
            kept = []
            for s in plan.steps:
                drop = False
                t = _txt(s.target_item_index) if s.target_item_index is not None else ""
                for rx in banned:
                    if s.action in ("click", "double_click", "right_click") and rx.search(t):
                        drop = True; break
                if not drop:
                    kept.append(s)
            plan.steps = kept
            return plan

        def _verify_plan_against_constraints(plan_json: str, hard_constraints: str) -> tuple[bool, str]:
            verifier_sys = f"""
                You judge ONLY whether a UI plan violates the bans below.
                - DO NOT check whether the final goal is achieved yet; intermediate steps are expected.
                - Mark VIOLATION only if a step directly conflicts with a ban (e.g., clicking a forbidden label, submitting when disallowed).

                BANS:
                \"\"\"{hard_constraints}\"\"\"
    """.strip()

            out = generate_text(
                prompt=f"PLAN JSON:\n{plan_json}\n\nAnswer with EXACTLY: OK <reason>  OR  VIOLATION <what and why>",
                system_prompt=verifier_sys,
                model="gpt-5-mini",
                reasoning_level="low",
            )
            s = str(out).strip()
            return (True, s) if s.upper().startswith("OK") else (False, s)

        # ---- Navigation instrumentation (reuse your prior robust code) ----
        def _install_spa_hook_once():
            if getattr(self, "_spa_hook_installed", False):
                return
            self.page.add_init_script("""
    (() => {
    try {
        if (window.__spaNavInstalled) return;
        window.__spaNavInstalled = true;
        window.__spaNavCount = 0;
        const bump = () => { window.__spaNavCount = (window.__spaNavCount||0) + 1; };
        const wrap = (name) => {
        const orig = history[name];
        history[name] = function() {
            const rv = orig.apply(this, arguments);
            try { dispatchEvent(new Event('app:navigation')); bump(); } catch(e){}
            return rv;
        }
        }
        wrap('pushState'); wrap('replaceState');
        addEventListener('popstate', () => { bump(); });
        addEventListener('hashchange', () => { bump(); });
    } catch(e){}
    })();
            """)
            self._spa_hook_installed = True

        def _spa_nav_count() -> int:
            try:
                return int(self.page.evaluate("() => window.__spaNavCount || 0"))
            except Exception:
                return 0

        def _get_document_dimensions() -> tuple[int, int]:
            try:
                doc_h = int(self.page.evaluate("() => document.documentElement ? document.documentElement.scrollHeight : 0"))
                vp_h = int(self.page.evaluate("() => window.innerHeight || 0"))
                return doc_h, vp_h
            except Exception:
                return 0, 0

        def _flatten_ignore(elem_map: dict[int, List[DetectedItem]]) -> list[dict]:
            out = []
            for run_idx, items in elem_map.items():
                for it in items:
                    out.append({"box_2d": it.box_2d, "why": f"ignored_from_run_{run_idx}"})
            return out

        def _center_from_box(box: list[int], meta: ViewportMeta) -> tuple[int, int]:
            ymin, xmin, ymax, xmax = box
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [ymin, xmin, ymax, xmax]):
                return 0, 0
            ymin = max(0, min(1000, int(ymin)))
            xmin = max(0, min(1000, int(xmin)))
            ymax = max(0, min(1000, int(ymax)))
            xmax = max(0, min(1000, int(xmax)))
            cx = int((xmin + xmax) * 0.5 / 1000.0 * meta.width)
            cy = int((ymin + ymax) * 0.5 / 1000.0 * meta.height)
            return max(0, min(meta.width - 1, cx)), max(0, min(meta.height - 1, cy))

        def _is_click_target_clear(x: int, y: int) -> bool:
            try:
                return bool(self.page.evaluate("""
                (x, y) => {
                    try {
                    if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
                    const els = document.elementsFromPoint(x, y);
                    if (!els || !els.length) return false;
                    const top = els[0];
                    const st = getComputedStyle(top);
                    return st.pointerEvents !== 'none' && st.visibility !== 'hidden' && st.display !== 'none';
                    } catch { return false; }
                }
                """, x, y))
            except Exception:
                return False

        def _clamp_detection(plan: "ModelPlan"):
            try:
                for it in plan.detection_result.items:
                    if getattr(it, "box_2d", None):
                        it.box_2d = [max(0, min(1000, int(v))) if isinstance(v, (int, float)) and math.isfinite(v) else 0 for v in it.box_2d]
                    try:
                        it.confidence = max(0.0, min(1.0, float(it.confidence)))
                    except Exception:
                        it.confidence = 0.5
            except Exception:
                pass
            for s in plan.steps or []:
                if s.to_ratio is not None:
                    try:
                        s.to_ratio = float(s.to_ratio)
                        s.to_ratio = 0.0 if s.to_ratio < 0 else 1.0 if s.to_ratio > 1 else s.to_ratio
                    except Exception:
                        s.to_ratio = None
                for fld in ("x", "y", "x2", "y2"):
                    v = getattr(s, fld)
                    if v is not None:
                        try:
                            setattr(s, fld, int(v))
                        except Exception:
                            setattr(s, fld, None)

        def _prepare_steps(plan: "ModelPlan", meta) -> list["Step"]:
            items = plan.detection_result.items if plan.detection_result else []
            prepared = []
            for s in plan.steps or []:
                if s.action in ("click", "double_click", "right_click"):
                    if s.target_item_index is not None and 0 <= s.target_item_index < len(items):
                        cx, cy = _center_from_box(items[s.target_item_index].box_2d, meta)
                        s.x = s.x or cx
                        s.y = s.y or cy
                    if s.x is not None and s.y is not None and not _is_click_target_clear(s.x, s.y):
                        for dx, dy in ((0, -10), (0, 10), (-10, 0), (10, 0)):
                            if _is_click_target_clear(s.x + dx, s.y + dy):
                                s.x += dx; s.y += dy
                                break
                        else:
                            try:
                                doc_h = int(self.page.evaluate("document.documentElement.scrollHeight"))
                                abs_y = meta.scroll_y + (s.y or 0)
                                ratio = min(1.0, max(0.0, abs_y / max(1, doc_h)))
                                prepared.append(type(s)(action="scroll", to_ratio=ratio))
                                prepared.append(type(s)(action="wait", ms=300))
                            except Exception:
                                prepared.append(type(s)(action="wait", ms=200))
                prepared.append(s)
            return prepared

        def _fingerprint_plan(plan: "ModelPlan") -> str:
            return "|".join(f"{s.action}:{s.target_item_index}:{s.x}:{s.y}:{(s.text or '')[:24]}" for s in (plan.steps or []))

        # Initialize cookie handler
        cookie_handler = CookieHandler(self.page, max_attempts=2)

        # ---- State ----
        _install_spa_hook_once()
        runs = 0
        elements_to_ignore: dict[int, List[DetectedItem]] = {}
        last_url = self.page.url
        last_spa = _spa_nav_count()
        total_actions = 0
        last_plan_fingerprint = None
        repeat_plan_count = 0
        last_shot_hash = None
        same_shot_count = 0
        
        self._goal_start_url = self.page.url
        
        # ---- Loop ----
        while runs < attempts:
            current_url = self.page.url
            current_spa = _spa_nav_count()
            
            if self.goals_met(goals):
                print("✅ Completion detected after navigation reset. Stopping.")
                self.page.pause()
                return

            # Reset on navigation (don't exit)
            if (current_url != last_url or current_spa != last_spa) and runs > 0:
                print(f"🔄 Navigation detected. Resetting. {last_url} → {current_url} / SPA {last_spa} → {current_spa}")
                elements_to_ignore.clear()
                runs = 0
                last_plan_fingerprint = None
                repeat_plan_count = 0
                total_actions = 0
                last_url = current_url
                last_spa = current_spa
                self.page.wait_for_load_state("domcontentloaded")
                self.page.wait_for_timeout(300)
                continue

            runs += 1
            cookie_handler.handle_cookies_with_dom()

            # Scroll setup
            print("Scrolling to top")
            try: self.page.evaluate("window.scrollTo(0, 0)")
            except Exception: pass
            self.page.wait_for_timeout(200)

            # Document dimensions
            doc_h, vp_h = _get_document_dimensions()
            if doc_h == 0 or vp_h == 0:
                doc_h, vp_h = 2000, 900

            if (runs - 1) in elements_to_ignore and len(elements_to_ignore[runs - 1]) > 1:
                top_y = min(it.box_2d[0] for it in elements_to_ignore[runs - 1])
                bot_y = max(it.box_2d[2] for it in elements_to_ignore[runs - 1])
                start_y = max(0, int(top_y * doc_h / 1000) - 200)
                end_y = min(doc_h, int(bot_y * doc_h / 1000) + 200)
                website_height = max(0, end_y - start_y)
                scroll_start = start_y
            else:
                website_height = doc_h
                scroll_start = 0

            step = max(1, int(vp_h * VIEWPORT_OVERLAP))
            old_url_for_loop = current_url
            old_spa_for_loop = current_spa
            navigated_mid_run = False

            y = scroll_start
            while y < scroll_start + website_height:
                if self.page.url != old_url_for_loop or _spa_nav_count() != old_spa_for_loop:
                    navigated_mid_run = True
                    break

                self.page.wait_for_load_state("domcontentloaded")
                try: self.page.evaluate(f"window.scrollTo(0, {y})")
                except Exception: pass
                self.page.wait_for_timeout(250)

                # Capture & build prompts
                meta, screenshot = self._capture_meta_and_screenshot()
                shot_hash = hashlib.md5(screenshot).hexdigest()
                same_shot_count = same_shot_count + 1 if shot_hash == last_shot_hash else 0
                last_shot_hash = shot_hash
                
                # --- anti-stuck using MAX_SAME_SHOTS ---
                if same_shot_count >= MAX_SAME_SHOTS:
                    print(f"[anti-stuck] Same viewport seen {same_shot_count+1} times; nudging state.")
                    same_shot_count = 0  # reset so we don't loop on every frame

                    # Tier 1: small jiggle scroll
                    try:
                        self.page.evaluate("window.scrollBy(0, Math.max(60, Math.floor(window.innerHeight*0.15)))")
                        self.page.wait_for_timeout(250)
                    except Exception:
                        pass

                    # Re-snapshot; if still identical, escalate
                    meta2, shot2 = self._capture_meta_and_screenshot()
                    if hashlib.md5(shot2).hexdigest() == shot_hash:
                        # Tier 2: tiny up/down jiggle to break sticky observers
                        try:
                            self.page.evaluate("window.scrollBy(0, -40)")
                            self.page.wait_for_timeout(150)
                            self.page.evaluate("window.scrollBy(0, 80)")
                            self.page.wait_for_timeout(200)
                        except Exception:
                            pass

                        # Tier 3: still stuck? skip ahead in the scan window
                        y += max(1, int(vp_h * 0.9))
                        continue  # restart inner loop with advanced y


                do_not_touch = _flatten_ignore(elements_to_ignore) if elements_to_ignore else []
                active_adapter = _active_adapter()
                # Merge preferences with hints (profile/assets/auth/dictionary/policies)
                eff_prefs = self._merge_preferences(getattr(self, "preferences", {}) or {}, hints)
                hint_snapshot = self._snapshot_from_hints(hints, active_adapter)
                hard_constraints = self._build_hard_constraints(meta_instructions, hints, active_adapter)

                system_prompt = _unified_system_prompt(
                    target_prompt=prompt,
                    meta=meta,
                    preferences=eff_prefs,
                    do_not_touch=do_not_touch,
                    hard_constraints=hard_constraints,
                    hint_snapshot=hint_snapshot,
                )
                user_prompt = "Plan now. Return ONLY the JSON for ModelPlan.\n\nBANS:\n" + (hard_constraints or "")

                # Generate plan
                plan = None
                for r in range(ONEPASS_RETRIES):
                    try:
                        plan = generate_model(
                            prompt=user_prompt,
                            model_object_type=ModelPlan,
                            system_prompt=system_prompt,
                            model=self.model,
                            reasoning_level="none",
                            image=screenshot,
                        )
                        break
                    except Exception as e:
                        print(f"[one-pass] schema/parse retry {r+1}/{ONEPASS_RETRIES}: {e}")
                        self.page.wait_for_timeout(300)

                if not plan:
                    # lazy-load height growth
                    try:
                        new_doc_h = int(self.page.evaluate("() => document.documentElement ? document.documentElement.scrollHeight : 0"))
                        if new_doc_h > doc_h * HEIGHT_GROWTH_FACTOR:
                            website_height += (new_doc_h - doc_h)
                            doc_h = new_doc_h
                    except Exception:
                        pass
                    y += step
                    continue

                _clamp_detection(plan)

                # Verifier micro-check
                try:
                    plan_json = plan.model_dump_json() if hasattr(plan, "model_dump_json") else json.dumps(plan, default=lambda o: o.__dict__)
                except Exception:
                    plan_json = json.dumps(plan, default=lambda o: getattr(o, "__dict__", str(o)))
                ok, msg = _verify_plan_against_constraints(plan_json, hard_constraints)
                if not ok:
                    print(f"[constraints] violation: {msg} → regenerating once")
                    regen_sys = system_prompt  # keep same but constraints already inlined
                    try:
                        plan: ModelPlan = generate_model(
                            prompt=user_prompt + f"\n\nNOTE: Previous plan violated constraints: {msg}\nFix and replan.",
                            model_object_type=ModelPlan,
                            system_prompt=regen_sys,
                            model=self.model,
                            reasoning_level="none",
                            image=screenshot,
                        )
                        _clamp_detection(plan)
                        plan_json = plan.model_dump_json() if hasattr(plan, "model_dump_json") else json.dumps(plan, default=lambda o: o.__dict__)
                        ok2, _ = _verify_plan_against_constraints(plan_json, hard_constraints)
                        if not ok2:
                            plan.steps = []
                    except Exception:
                        plan.steps = []

                # Local ban rules (from constraints + adapter)
                rules = _compile_meta_rules(hard_constraints, active_adapter)
                plan = _enforce_rules_locally(plan, rules, plan.detection_result.items if plan and plan.detection_result else [])

                plan = self._apply_fill_policy(plan, meta, (hints.goal_progress or ProgressPolicy()).fill_only)


                # Prepare steps
                plan.steps = _prepare_steps(plan, meta)

                # Cull invalid indices
                n_items = len(plan.detection_result.items) if plan.detection_result else 0
                valid_steps, used_indices = [], set()
                for s in plan.steps or []:
                    if s.target_item_index is None or (0 <= s.target_item_index < n_items):
                        valid_steps.append(s)
                        if s.target_item_index is not None:
                            used_indices.add(s.target_item_index)
                plan.steps = valid_steps

                # Repeat-plan guard
                fp = _fingerprint_plan(plan)
                repeat_plan_count = repeat_plan_count + 1 if fp == last_plan_fingerprint else 0
                last_plan_fingerprint = fp
                if repeat_plan_count >= MAX_REPEAT_PLAN or same_shot_count >= MAX_SAME_SHOTS:
                    print("Plan repeated without effect; stopping to avoid loop.")
                    break

                if plan.steps:
                    url_before = self.page.url
                    spa_before = _spa_nav_count()
                    self._exec_plan(plan, meta)
                    total_actions += len(plan.steps)
                    
                    if self.goals_met(goals):
                        print("✅ Completion detected after navigation reset. Stopping.")
                        self.page.pause()
                        return

                    # Ignore used elements next run
                    if used_indices:
                        if runs not in elements_to_ignore:
                            elements_to_ignore[runs] = []
                        for idx in sorted(used_indices):
                            try:
                                elements_to_ignore[runs].append(plan.detection_result.items[idx])
                            except Exception:
                                pass

                    if self.page.url != url_before or _spa_nav_count() != spa_before:
                        navigated_mid_run = True
                        break

                    if total_actions > MAX_ACTIONS:
                        print("Hit global action cap; stopping.")
                        break
                else:
                    # extend scan window if height grew
                    try:
                        new_doc_h = int(self.page.evaluate("() => document.documentElement ? document.documentElement.scrollHeight : 0"))
                        if new_doc_h > doc_h * HEIGHT_GROWTH_FACTOR:
                            website_height += (new_doc_h - doc_h)
                            doc_h = new_doc_h
                    except Exception:
                        pass

                y += step

            if navigated_mid_run:
                last_url = self.page.url
                last_spa = _spa_nav_count()
                elements_to_ignore.clear()
                runs = 0
                last_plan_fingerprint = None
                repeat_plan_count = 0
                total_actions = 0
                self.page.wait_for_load_state("domcontentloaded")
                self.page.wait_for_timeout(300)
                if self.goals_met(goals):
                    print("✅ Completion detected after navigation reset. Stopping.")
                    self.page.pause()
                    return
                continue

            # (Optional) you can insert your "nudge" block here as in your previous version
            # using the same hard_constraints + hints snapshot, then loop.

            last_url = self.page.url
            last_spa = _spa_nav_count()

        print("Finished executing compressed phrase.")
 
    def _canon(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())
        
    def run(
        self,
        prompt: Optional[str] = None,
        *,
        subprompt_hints: Optional[Mapping[str, Union[UXHints, Dict[str, Any]]]] = None,
        strict: bool = False,
    ):
        """
        Execute a prompt broken into sub-prompts, OR execute an explicit list of
        sub-prompts with supplied UXHints.

        Args:
            prompt: Optional full instruction to decompose. If omitted/blank, we use
                    the keys of `subprompt_hints` (in insertion order) as sub-prompts.
            subprompt_hints: Optional mapping {sub-prompt -> UXHints | dict}. If a
                    hint is not provided for a sub-prompt (when `prompt` is used),
                    we'll infer one unless `strict=True`.
            strict: If True, require a manual hint for every sub-prompt (when
                    `prompt` is provided). If prompt is omitted, all sub-prompts
                    MUST come from `subprompt_hints`.
        """
        # Build manual hints map (canonicalized)
        manual_map: Dict[str, UXHints] = {}
        if subprompt_hints:
            for k, v in subprompt_hints.items():
                if isinstance(v, dict):
                    try:
                        v = UXHints.model_validate(v)
                    except ValidationError as e:
                        raise ValueError(f"Invalid UXHints for '{k}': {e}") from e
                elif not isinstance(v, UXHints):
                    raise TypeError(f"Hint for '{k}' must be UXHints or dict, got {type(v).__name__}")
                manual_map[self._canon(k)] = v

        # Decide sub-prompts source
        if prompt and prompt.strip():
            sub_prompts = self.parse_task_into_subtasks(prompt).sub_tasks
        else:
            if not subprompt_hints:
                raise ValueError("Provide 'prompt' or 'subprompt_hints'.")
            # Use keys in insertion order (Py3.7+ dicts preserve it)
            sub_prompts = list(subprompt_hints.keys())

        for sp in sub_prompts:
            print(f"\n--- Attempt {sp} ---")
            key = self._canon(sp)

            if manual_map:
                if key in manual_map:
                    hints = manual_map[key]
                    print(f"Using MANUAL hints for '{sp}'.")
                else:
                    if strict or not (prompt and prompt.strip()):
                        # No fallback if strict OR when running without a prompt
                        raise ValueError(f"No manual hints provided for sub-prompt: '{sp}'")
                    # Fall back to inference only when a prompt was used
                    hints = self.infer_hints_for_subprompt(sp, profile=None, dictionary=None)
                    print(f"Using INFERRED hints for '{sp}'.")
            else:
                # No manual hints at all → infer
                hints = self.infer_hints_for_subprompt(sp, profile=None, dictionary=None)
                print(f"Using INFERRED hints for '{sp}'.")

            # Safety: if anything snuck in as a dict, validate to UXHints
            if isinstance(hints, dict):
                try:
                    hints = UXHints.model_validate(hints)
                except ValidationError as e:
                    raise ValueError(f"Invalid hints for '{sp}': {e}") from e

            print(f"Hints: {hints.model_dump_json(indent=2)}")

            self._exec_compressed_phrase(
                prompt=sp,
                meta_instructions="Avoid destructive or final actions unless explicitly instructed.",
                attempts=60,
                hints=hints,
            )
     
    def find_target_in_detections(self, detection_result: DetectionResult, target_description: str) -> Optional[int]:
        """Find the index of the target item in the detection result."""
        sys_prompt = f"""
        Your job is to find the index of the target item in the detection result.
        The target item is {target_description}.
        The detection result is {detection_result.model_dump_json(indent=2)}.
        
        The first item has index 0.
        
        If the target item is not found, return -1.
        """
        user_prompt = """
        Return the index of the target item.
        """
        return_value = generate_text(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            model="gpt-5-nano",
            reasoning_level="high",
        )
        return int(return_value.strip())

# -------------------------- Capture --------------------------
    def _capture_meta_and_screenshot(self) -> Tuple[ViewportMeta, bytes]:
        vp = self.page.viewport_size
        width, height = int(vp["width"]), int(vp["height"])
        
        # Check if page context is valid before proceeding
        if not self._is_page_context_valid():
            print("Warning: Page context not valid, using fallback values")
            # Return fallback metadata with safe values
            meta = ViewportMeta(
                width=width,
                height=height, 
                dpr=1.0, 
                scroll_x=0,
                scroll_y=0, 
                ss_pixel_w=width, 
                ss_pixel_h=height, 
                css_scale=1.0, 
                doc_width=width, 
                doc_height=height)
            img = self.page.screenshot(full_page=False)  # viewport-only
            return meta, img
        
        try:
            dpr = float(self.page.evaluate("window.devicePixelRatio"))
            scroll_x = int(self.page.evaluate("window.scrollX"))
            scroll_y = int(self.page.evaluate("window.scrollY"))
            
            # Safely get document dimensions with error handling
            doc_dimensions = self.page.evaluate("""
                () => {
                    try {
                        if (!document || !document.documentElement) {
                            return null;
                        }
                        return {
                            clientWidth: document.documentElement.clientWidth,
                            clientHeight: document.documentElement.clientHeight,
                            scrollWidth: document.documentElement.scrollWidth,
                            scrollHeight: document.documentElement.scrollHeight
                        };
                    } catch (e) {
                        return null;
                    }
                }
            """)
            
            if doc_dimensions is None:
                print("Warning: Could not get document dimensions, using fallback values")
                ss_pixel_w = width
                ss_pixel_h = height
                doc_width = width
                doc_height = height
            else:
                ss_pixel_w = int(doc_dimensions.get('clientWidth', width))
                ss_pixel_h = int(doc_dimensions.get('clientHeight', height))
                doc_width = int(doc_dimensions.get('scrollWidth', width))
                doc_height = int(doc_dimensions.get('scrollHeight', height))
            
            css_scale = float(ss_pixel_w / width) if width > 0 else 1.0
            
        except Exception as e:
            print(f"Error getting page metadata: {e}, using fallback values")
            dpr = 1.0
            scroll_x = 0
            scroll_y = 0
            ss_pixel_w = width
            ss_pixel_h = height
            css_scale = 1.0
            doc_width = width
            doc_height = height
        
        if self.viewport_lock and abs(dpr - 1.0) > 1e-6:
            raise RuntimeError(
                f"DPR must be 1.0 for coordinate clarity; got {dpr}. Set device_scale_factor=1 and zoom=100%."
            )

        meta = ViewportMeta(
            width=width,
            height=height, 
            dpr=dpr, 
            scroll_x=scroll_x,
            scroll_y=scroll_y, 
            ss_pixel_w=ss_pixel_w, 
            ss_pixel_h=ss_pixel_h, 
            css_scale=css_scale, 
            doc_width=doc_width, 
            doc_height=doc_height)
        img = self.page.screenshot(full_page=False)  # viewport-only
        return meta, img
# -------------------------- Execution --------------------------
    def verify_target_item_in_detections(self, detection_result: DetectionResult, step: Step, target_description: str) -> int:
        """Verify that the target item is in the detection result."""
        proposed_index = step.target_item_index
        proposed_item_description = detection_result.items[step.target_item_index].description
        
        verified_index = self.find_target_in_detections(detection_result, target_description)
        
        system_prompt = f"""
        Your job is to compare the proposed item description {proposed_item_description} with the target description {target_description}.
        and determine if they are the same
        If they are the same, return 1.
        If they are not the same, return 0.
        
        Example:
        Proposed item description: "First Name"
        Target description: "First Name"
        Return: 1
        
        Proposed item description: "First Name"
        Target description: "Last Name"
        Return: 0
        """
        user_prompt = """
        Return 1 if the proposed item description is the same as the target description, otherwise return 0.
        """
        
        return_value = generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model="gpt-5-nano",
            reasoning_level="high",
        )
        
        if return_value.strip() == "1":
            return proposed_index
        else:
            return verified_index
    
    def _exec_plan(self, plan: ModelPlan, meta: ViewportMeta) -> None:
        detection_result = plan.detection_result
        print(f"\n🎯 EXECUTING PLAN: {len(plan.steps)} steps")
        print(f"📊 Viewport: {meta.width}x{meta.height}px, Scroll: ({meta.scroll_x}, {meta.scroll_y})")
        
        for i, step in enumerate(plan.steps):
            print(f"\n--- STEP {i+1}/{len(plan.steps)} ---")
            print(f"🔧 Action: {step.action}")
            if step.text:
                print(f"📝 Text: '{step.text}'")
            if step.target_item_index is not None:
                print(f"🎯 Target index: {step.target_item_index}")
            
            a = step.action
            target_item = None
            
            # Handle coordinate-based actions
            if a in ("click", "type", "press", "double_click", "right_click", "move", "drag"):
                x, y = None, None
                
                # Try to use detected item first
                if step.target_item_index is not None:
                    if 0 <= step.target_item_index < len(detection_result.items):
                        target_item: DetectedItem = detection_result.items[step.target_item_index]
                        if target_item.viewport_data.scroll_x is not None and target_item.viewport_data.scroll_y is not None:
                            print(f"Scrolling to {target_item.viewport_data.scroll_x}, {target_item.viewport_data.scroll_y}")
                            self.page.wait_for_timeout(5000)
                            self.page.evaluate(f"window.scrollTo({target_item.viewport_data.scroll_x}, {target_item.viewport_data.scroll_y})")
                            time.sleep(2)
                        print(f"Target index: {step.target_item_index}, Target item: {target_item.description}")

                        # Verify the target item is in the detection result
                        verified_index = self.verify_target_item_in_detections(detection_result, step, target_item.description)
                        
                        print(f"Verified index: {verified_index} for {target_item.description}")
                        
                        if verified_index == -1:
                            print("Could not find the target item in the detection result, retrying...")
                            self.page.evaluate("window.scrollBy(0, 100)")
                            time.sleep(2)
                            return self._exec_plan(plan, meta)
                        
                        step.target_item_index = verified_index
                    else:
                        print(f"WARNING: Invalid target_item_index {step.target_item_index}")
                        # Verify the target item is in the detection result
                        verified_index = self.find_target_in_detections(detection_result, step.action_note)
                        if verified_index == -1:
                            print("Could not find the target item in the detection result, retrying...")
                            self.page.evaluate("window.scrollBy(0, 100)")
                            time.sleep(2)
                            return self._exec_plan(plan, meta)
                        
                        step.target_item_index = verified_index
                    
                    target_item = detection_result.items[step.target_item_index]
                    
                        
                    x, y = _get_item_center_pixels(target_item, meta.width, meta.height)
                    print(f"📍 Item box: {target_item.box_2d}, Center: ({x}, {y})")
                    print(f"📋 Target description: {target_item.description}")
                    print(f"🏷️  Item type: {target_item.item_type}, Input type: {target_item.input_type}")
                
                # Fallback to direct coordinates
                if x is None and step.x is not None and step.y is not None:
                    x = _clamp(int(step.x), 0, meta.width - 1)
                    y = _clamp(int(step.y), 0, meta.height - 1)
                    print(f"📍 Using direct coordinates: ({x}, {y})")
                
                # Skip if no valid coordinates
                if (x is None or y is None) and a != "press":
                    print(f"⚠️  No valid coordinates for {a} action, skipping")
                    continue
                
                if x is not None and y is not None:
                    # Clamp coordinates to viewport
                    x = _clamp(x, 0, meta.width - 1)
                    y = _clamp(y, 0, meta.height - 1)
                    print(f"🎯 Final coordinates: ({x}, {y})")

            if a == "click":
                self.page.mouse.click(x, y, button=step.button or "left")
                if target_item is None:
                    continue
                        
                if target_item.triggers_file_input:
                    self.page.pause()
                    
                print(f"✅ Clicked at {x}, {y}")
            elif a == "double_click":
                self.page.mouse.dblclick(x, y)
                print(f"✅ Double clicked at {x}, {y}")
            elif a == "right_click":
                self.page.mouse.click(x, y, button="right")
                print(f"✅ Right clicked at {x}, {y}")
            elif a == "move":
                self.page.mouse.move(x, y)
                print(f"✅ Moved to {x}, {y}")
            elif a == "drag":
                # Handle drag end coordinates
                x2, y2 = None, None
                if step.target_item_index is not None and step.x2 is None:
                    # Use same item for end point (not typical, but possible)
                    x2, y2 = x, y
                elif step.x2 is not None and step.y2 is not None:
                    x2 = _clamp(int(step.x2), 0, meta.width - 1)
                    y2 = _clamp(int(step.y2), 0, meta.height - 1)
                else:
                    x2, y2 = x, y  # Default to same position
                
                self.page.mouse.move(x, y)
                self.page.mouse.down()
                steps = max(abs(x2 - x), abs(y2 - y)) // 50 + 10
                self.page.mouse.move(x2, y2, steps=steps)
                self.page.mouse.up()
                print(f"✅ Dragged from {x}, {y} to {x2}, {y2}")
            elif a == "type":
                if step.text:
                    self.page.mouse.click(x, y, button="left")
                    print(f"✅ Clicked at {x}, {y} to type {step.text}")
                    time.sleep(1)
                    self.page.keyboard.type("", delay=100)
                    self.page.keyboard.type(step.text, delay=100)
                    print(f"✅ Typed: {step.text}")
            elif a == "press":
                if step.key:
                    self.page.keyboard.press(step.key)
                    print(f"✅ Pressed key: {step.key}")
            elif a == "scroll":
                # r = float(step.to_ratio or 0.0)
                # r = max(0.0, min(1.0, r))
                # scroll_y = int(r * meta.doc_height)
                # print(f"📜 Scrolling to ratio {r:.2f} (y={scroll_y}px)")
                # self.page.evaluate(
                #     """
                #     (r) => {
                #       const max = document.documentElement.scrollHeight - window.innerHeight;
                #       window.scrollTo(0, Math.max(0, Math.min(max, r*max)));
                #     }
                #     """,
                #     r,
                # )
                # print(f"✅ Scrolled to ratio {r:.2f} (y={scroll_y}px)")
                continue
            elif a == "wait":
                ms = int(step.ms or 200)
                print(f"⏳ Waiting for {ms}ms...")
                self.page.wait_for_timeout(ms)
                print(f"✅ Waited for {ms}ms")
            else:
                print(f"⚠️  Unknown action '{a}' - skipping")
                continue


      
    
    
    def _is_page_context_valid(self) -> bool:
        """
        Checks if the page context is still valid and accessible.
        """
        try:
            if not self.page or not hasattr(self.page, 'url'):
                return False
            
            # Try to access a basic page property
            current_url = self.page.url
            if not current_url or current_url == "about:blank":
                return False
            
            # Try a simple evaluation to test context
            result = self.page.evaluate("() => document.readyState")
            return result in ["loading", "interactive", "complete"]
            
        except Exception:
            return False
    
if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        page = ctx.new_page()
        # page.goto("https://www.sumup.com/careers/positions/london-england-united-kingdom/ios/senior-ios-engineer-global-bank/8048304002/?gh_jid=8048304002&gh_src=jn5gvww32us", wait_until="domcontentloaded")
        page.goto("https://news.ycombinator.com/", wait_until="domcontentloaded")
        time.sleep(3)

        vc = VisionCoordinator(
            page,
            model="gemini-2.5-flash",
            reasoning_level="none",
            image_detail="low",
            preferences={
                "username": "standard_user",
                "password": "secret_sauce",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "1234567890",
            }
        )
        
        vc.run(
            prompt="navigate to the first item in the list",
        )