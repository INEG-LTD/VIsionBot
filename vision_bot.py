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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from playwright.sync_api import Page
from ai_utils import generate_model, generate_text
from cookie_handler import CookieHandler
from text_utils import TextUtils
from form_utils import FormUtils
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
    # Removed duplicate: viewport_data: ViewportMeta
    
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

    def is_effectively_empty(self) -> bool:
        return not any([
            bool(self.headings),
            bool(self.url_contains),
            bool(self.title_contains),
            bool(self.body_contains),
            bool(self.aria_active_tab),
            bool(self.hostname_in),
            bool(self.path_contains),
        ])

    @model_validator(mode="after")
    def _fallback_when_empty(self):
        """
        If the StopWhen is completely empty (LLM didn’t provide any filters),
        default to a change-based detector: require_url_change and at least 1 strong signal.
        This prevents silent no-op goals.
        """
        try:
            if self.is_effectively_empty():
                if not self.require_url_change:
                    self.require_url_change = True
                if not isinstance(self.min_strong_signals, int) or self.min_strong_signals < 1:
                    self.min_strong_signals = 1
                if self.match_mode not in ("any", "all"):
                    self.match_mode = "any"
        except Exception:
            pass
        return self



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
    LIST_ITEM_OPENED = "list_item_opened"
    FORM_COMPLETED = "form_completed"
    LOGIN_COMPLETED = "login_completed"
    ELEMENT_VISIBLE = "element_visible"
    ELEMENT_GONE = "element_gone"
    LIST_COUNT = "list_count"
    FILTER_APPLIED = "filter_applied"
    FIELD_VALUE_SET = "field_value_set"
    UPLOAD_ATTACHED = "upload_attached"
    NEW_TAB_OPENED = "new_tab_opened"
    REPEAT_UNTIL = "repeat_until"
    ANY_OF = "any_of"                           # meta-goal: any child goal suffices
    ALL_OF = "all_of"                           # meta-goal: all child goals required
    SURFACE_GONE = "surface_gone"


class GoalBase(BaseModel):
    type: GoalKind
    
class ListLocator(BaseModel):
    """How to find the list container."""
    region: Optional[SectionIn] = None           # header|content|sidebar|modal|...
    heading_contains: List[str] = Field(default_factory=list)  # text in heading immediately above the list
    aria_label_contains: List[str] = Field(default_factory=list)
    container_kind: Literal["auto","ul","ol","table","grid","feed"] = "auto"
    min_items: int = 3                           # must look like a real list

class ItemSpec(BaseModel):
    """
    How to pick an item inside the chosen list container.
    If 'nth' or 'position' are provided, index selection is used.
    Otherwise, we use content-based selection with these filters.
    """
    # Index-based (kept)
    position: Literal["first", "last"] = "first"
    nth: Optional[int] = None

    # Content-based
    text_any: List[str] = Field(default_factory=list)   # pass if ANY token appears in item label
    text_all: List[str] = Field(default_factory=list)   # pass if ALL tokens appear
    regex: Optional[str] = None                         # JS RegExp source (case-insensitive)
    url_contains: List[str] = Field(default_factory=list)
    domain_in: List[str] = Field(default_factory=list)  # e.g., ["nytimes.com","openai.com"]

    # General constraints
    exclude_text: List[str] = Field(default_factory=lambda: ["sponsored","ad","promotion"])
    require_link: bool = True
    require_visible: bool = True

    # Tie breaking when multiple match
    tie_breaker: Literal["first", "last", "highest_score"] = "highest_score"

    # Optional semantic selection: only used if provided
    semantic_query: Optional[str] = None                # short phrase to describe the target item
    top_k_semantic: int = 12                            # candidates to send to LLM if semantic_query set

class ListItemOpenedGoal(GoalBase):
    type: Literal[GoalKind.LIST_ITEM_OPENED] = Field(default=GoalKind.LIST_ITEM_OPENED)
    locator: ListLocator = Field(default_factory=ListLocator)
    item: ItemSpec = Field(default_factory=ItemSpec)
    require_url_change: bool = True              # opening should navigate (set False if you only want focus/selection)

# ----------- Inserted: Helper model & new goal classes -----------
class ElementMatch(BaseModel):
    text_any: List[str] = Field(default_factory=list)
    role_any: List[str] = Field(default_factory=list)  # e.g. "button","link","heading"
    near_heading: List[str] = Field(default_factory=list)
    region: Optional[SectionIn] = None  # narrow to header/content/sidebar/modal

class ElementVisibleGoal(GoalBase):
    type: Literal[GoalKind.ELEMENT_VISIBLE] = Field(default=GoalKind.ELEMENT_VISIBLE)
    match: ElementMatch
    min_count: int = 1

class ElementGoneGoal(GoalBase):
    type: Literal[GoalKind.ELEMENT_GONE] = Field(default=GoalKind.ELEMENT_GONE)
    match: ElementMatch

class ListCountGoal(GoalBase):
    type: Literal[GoalKind.LIST_COUNT] = Field(default=GoalKind.LIST_COUNT)
    locator: ListLocator = Field(default_factory=ListLocator)
    min_items: int = 20

class FilterAppliedGoal(GoalBase):
    type: Literal[GoalKind.FILTER_APPLIED] = Field(default=GoalKind.FILTER_APPLIED)
    chips_any: List[str] = Field(default_factory=list)
    facets_all: List[str] = Field(default_factory=list)
    url_params_contains: Dict[str, str] = Field(default_factory=dict)

class FieldValueSetGoal(GoalBase):
    type: Literal[GoalKind.FIELD_VALUE_SET] = Field(default=GoalKind.FIELD_VALUE_SET)
    label_any: List[str]
    want_value: str
    exact: bool = True

class UploadAttachedGoal(GoalBase):
    type: Literal[GoalKind.UPLOAD_ATTACHED] = Field(default=GoalKind.UPLOAD_ATTACHED)
    label_any: List[str] = Field(default_factory=lambda: ["resume","cv","attachment","cover letter"]) 
    require_preview_label: bool = True

class NewTabOpenedGoal(GoalBase):
    type: Literal[GoalKind.NEW_TAB_OPENED] = Field(default=GoalKind.NEW_TAB_OPENED)
    url_contains: List[str] = Field(default_factory=list)
    title_contains: List[str] = Field(default_factory=list)

class RepeatUntilGoal(GoalBase):
    type: Literal[GoalKind.REPEAT_UNTIL] = Field(default=GoalKind.REPEAT_UNTIL)
    goal: "Goal"
    max_iters: int = 10
    sleep_ms: int = 300

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

# ----------- Inserted: SurfaceSignature and SurfaceGoneGoal -----------
class SurfaceSignature(BaseModel):
    # Geometry & visual prominence
    rect: Dict[str, float]                     # {top,left,bottom,right,width,height}
    area_ratio: float = 0.0                    # element area / viewport area
    z: float = 0.0
    # Heuristics
    is_modal: bool = False
    is_banner: bool = False                    # fixed to top/bottom and wide
    # Content fingerprints (loose match)
    text_tokens: List[str] = Field(default_factory=list)
    button_texts: List[str] = Field(default_factory=list)

class SurfaceGoneGoal(GoalBase):
    type: Literal[GoalKind.SURFACE_GONE] = Field(default=GoalKind.SURFACE_GONE)
    signature: SurfaceSignature
    require_scroll_unlock: bool = True         # body overflow lock should clear for modals


Goal = Annotated[
    Union[
        PageReachedGoal,
        SubmissionConfirmedGoal,
        ListItemOpenedGoal,
        FormCompletedGoal,
        LoginCompletedGoal,
        ElementVisibleGoal,
        ElementGoneGoal,
        ListCountGoal,
        FilterAppliedGoal,
        FieldValueSetGoal,
        UploadAttachedGoal,
        NewTabOpenedGoal,
        SurfaceGoneGoal,
        RepeatUntilGoal,
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

class RunSpec(BaseModel):
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
        return list(self.goals or [])



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
    def _visible_match_js(self):
        # Small helper to reuse in element-visible/gone
        return """
        (match) => {
          const norm = s => (s||'').toLowerCase();
          const wantTxt = (match && match.text_any || []).map(norm);
          const wantRoles = (match && match.role_any || []).map(norm);
          const nearHeads = (match && match.near_heading || []).map(norm);
          const wantRegion = match && match.region && match.region.toLowerCase();
          const vis = el => { if(!el) return false; const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
            return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none'; };
          const txt = el => (el && (el.innerText||el.textContent)||'').replace(/\s+/g,' ').trim();
          const sectionOf = el => (
            (el.closest('header') && 'header') || (el.closest('footer') && 'footer') ||
            (el.closest('nav,[role="navigation"]') && 'navigation') ||
            (el.closest('aside,[role="complementary"]') && 'sidebar') ||
            (el.closest('dialog,[role="dialog"],[role="alertdialog"]') && 'modal') ||
            (el.closest('main,[role="main"]') && 'content') || 'content');
          const headingAbove = el => {
            let n=el; const top=el.getBoundingClientRect().top; let best='';
            while(n && n!==document.body){
              const prev=n.previousElementSibling; if(!prev){ n=n.parentElement; continue; }
              const heads=prev.querySelectorAll('h1,h2,h3,h4');
              for(const h of heads){ if(!vis(h)) continue; const d=top-h.getBoundingClientRect().bottom; if(d>=0 && d<=250){ const t=txt(h); if(t && t.length>best.length) best=t; } }
              n=prev;
            }
            return best.toLowerCase();
          };
          const roleSel = [];
          if(!wantRoles.length || wantRoles.includes('button')) roleSel.push('button,[role="button"],input[type="button"],input[type="submit"]');
          if(!wantRoles.length || wantRoles.includes('link')) roleSel.push('a[href]');
          if(!wantRoles.length || wantRoles.includes('heading')) roleSel.push('h1,h2,h3,h4,h5,h6');
          const nodes = Array.from(document.querySelectorAll(roleSel.join(',')));
          let hits=0; const examples=[];
          for(const el of nodes){
            if(!vis(el)) continue;
            if(wantRegion){ if(sectionOf(el)!==wantRegion) continue; }
            const t = txt(el).toLowerCase();
            if(wantTxt.length && !wantTxt.some(w=>t.includes(w))) continue;
            if(nearHeads.length){ const h=headingAbove(el); if(!nearHeads.some(w=>h.includes(w))) continue; }
            hits++; if(examples.length<5) examples.push(t.slice(0,120));
          }
          return {count:hits, examples};
        }
        """

    def evaluate_element_visible(self, g: "ElementVisibleGoal") -> "GoalResult":
        try:
            data = self.page.evaluate(self._visible_match_js(), g.match.model_dump())
            cnt = int(data.get("count") or 0)
            ok = cnt >= int(getattr(g, "min_count", 1) or 1)
            return GoalResult(status=GoalStatus.MET if ok else GoalStatus.UNMET,
                              score=1.0 if ok else min(0.9, cnt/ max(1,g.min_count)),
                              reason=f"{cnt} matching elements visible (need ≥{g.min_count})")
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"element_visible error: {e}")

    def _gone_strict_js(self):
        return """
        (match) => {
          const norm = s => (s||'').toLowerCase();
          const wantTxt = (match && match.text_any || []).map(norm);
          const wantRoles = (match && match.role_any || []).map(norm);
          const wantRegion = match && match.region && match.region.toLowerCase();

          const vis = el => { if(!el) return false; const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
            if (s.visibility==='hidden' || s.display==='none' || parseFloat(s.opacity||'1')<0.03) return false;
            if (s.pointerEvents==='none') return false;
            return r.width>0 && r.height>0 && r.bottom>0 && r.right>0 && r.top < (innerHeight||document.documentElement.clientHeight) && r.left < (innerWidth||document.documentElement.clientWidth);
          };
          const hiddenByAncestor = el => {
            for(let n=el;n && n!==document.body;n=n.parentElement){ const s=getComputedStyle(n);
              if (s.visibility==='hidden' || s.display==='none' || s.pointerEvents==='none') return true;
              if (s.clipPath && s.clipPath!=='none') return true;
            }
            return false;
          };
          const sectionOf = el => (
            (el.closest('header') && 'header') || (el.closest('footer') && 'footer') ||
            (el.closest('nav,[role="navigation"]') && 'navigation') ||
            (el.closest('aside,[role="complementary"]') && 'sidebar') ||
            (el.closest('dialog,[role="dialog"],[role="alertdialog"]') && 'modal') ||
            (el.closest('main,[role="main"]') && 'content') || 'content');

          const roleSel=[];
          if(!wantRoles.length || wantRoles.includes('button')) roleSel.push('button,[role="button"],input[type="button"],input[type="submit"]');
          if(!wantRoles.length || wantRoles.includes('link')) roleSel.push('a[href]');
          if(!wantRoles.length || wantRoles.includes('heading')) roleSel.push('h1,h2,h3,h4,h5,h6');
          const nodes = Array.from(document.querySelectorAll(roleSel.join(',')));

          const matchesText = (el) => {
            if(!wantTxt.length) return true;
            const t = (el.innerText||el.textContent||'').replace(/\s+/g,' ').trim().toLowerCase();
            return wantTxt.some(w=>t.includes(w));
          };

          const hitTest = (el) => {
            // Check if center point is clickable and whether el (or its ancestor) owns it
            const r = el.getBoundingClientRect();
            const cx = Math.floor(r.left + r.width/2);
            const cy = Math.floor(r.top + r.height/2);
            const topAt = document.elementFromPoint(cx, cy);
            if(!topAt) return false;
            return el===topAt || el.contains(topAt) || topAt.contains(el);
          };

          const cand = [];
          for(const el of nodes){
            if(!matchesText(el)) continue;
            const region = sectionOf(el);
            if(wantRegion && wantRegion!==region) continue;
            const v = vis(el);
            const ancHidden = hiddenByAncestor(el);
            const clickableHit = v && !ancHidden && hitTest(el);
            cand.push({v, ancHidden, clickableHit});
          }
          const stillVisible = cand.filter(c=>c.v && !c.ancHidden).length;
          const stillInteractive = cand.filter(c=>c.clickableHit).length;
          return {stillVisible, stillInteractive};
        }
        """;

    def evaluate_element_gone(self, g: "ElementGoneGoal") -> "GoalResult":
        try:
            data = self.page.evaluate(self._gone_strict_js(), g.match.model_dump())
            vis = int(data.get("stillVisible") or 0)
            hit = int(data.get("stillInteractive") or 0)
            ok = (vis == 0) and (hit == 0)
            score = 1.0 if ok else (0.0 if hit>0 else 0.2)  # if only non-interactive remnants remain, give tiny score
            reason = f"visible={vis}, interactiveHit={hit}"
            return GoalResult(status=GoalStatus.MET if ok else GoalStatus.UNMET, score=score, reason=reason)
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"element_gone error: {e}")

    def evaluate_list_count(self, g: "ListCountGoal") -> "GoalResult":
        try:
            data = self.page.evaluate("""
            (locator) => {
              const norm = s => (s||'').toLowerCase();
              const vis = el => { if(!el) return false; const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
                return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none'; };
              const txt = el => (el && (el.innerText||el.textContent)||'').replace(/\s+/g,' ').trim();
              const headingAbove = el => { let n=el; const top=el.getBoundingClientRect().top; let best='';
                while(n && n!==document.body){ const prev=n.previousElementSibling; if(!prev){ n=n.parentElement; continue; }
                  const heads=prev.querySelectorAll('h1,h2,h3,h4'); for(const h of heads){ if(!vis(h)) continue; const d=top-h.getBoundingClientRect().bottom; if(d>=0 && d<=250){ const t=txt(h); if(t && t.length>best.length) best=t; } } n=prev; }
                return best.toLowerCase(); };
              const sectionOf = el => ((el.closest('header')&&'header')||(el.closest('footer')&&'footer')||
                (el.closest('nav,[role="navigation"]')&&'navigation')||(el.closest('aside,[role="complementary"]')&&'sidebar')||
                (el.closest('dialog,[role="dialog"],[role="alertdialog"]')&&'modal')||(el.closest('main,[role="main"]')&&'content')||'content');

              const kinds=(locator && locator.container_kind)||'auto';
              const wantRegion=locator && locator.region && locator.region.toLowerCase();
              const wantHead=(locator && (locator.heading_contains||[])).map(norm);
              const wantAria=(locator && (locator.aria_label_contains||[])).map(norm);
              const containers=new Set();
              document.querySelectorAll('ul,ol,[role="list"],[role="feed"],table,[role="grid"]').forEach(c=>containers.add(c));
              document.querySelectorAll('main,[role="main"],body').forEach(root=>{
                const kids=Array.from(root.querySelectorAll('*')).filter(e=>e.children && e.children.length>3);
                for(const k of kids){ const ch=Array.from(k.children).filter(e=>vis(e)); const withLinks = ch.filter(c=>c.querySelector('a[href]'));
                  if(withLinks.length>=3) containers.add(k); }
              });
              const cands=[];
              containers.forEach(c=>{
                if(!vis(c)) return;
                if(kinds!=='auto'){
                  if(kinds==='ul' && c.tagName!=='UL') return;
                  if(kinds==='ol' && c.tagName!=='OL') return;
                  if(kinds==='table' && c.tagName!=='TABLE') return;
                  if(kinds==='grid' && !c.matches('table,[role="grid"]')) return;
                  if(kinds==='feed' && !c.matches('[role="feed"],[role="list"]')) return;
                }
                const region=sectionOf(c);
                if(wantRegion && region!==wantRegion) return;
                const head=headingAbove(c);
                const aria=norm(c.getAttribute('aria-label')||'');
                const smallRows = c.tagName==='TABLE' ? Array.from(c.querySelectorAll('tr')) : Array.from(c.children);
                const items = smallRows.filter(e=>vis(e));
                const score=(wantHead.length && wantHead.some(h=>head.includes(h))?2:0) + (wantAria.length && wantAria.some(h=>aria.includes(h))?1:0);
                cands.push({container:c, count:items.length, head, region, score});
              });
              if(!cands.length) return {count:0, reason:'no containers'};
              cands.sort((a,b)=> (b.score - a.score) || (b.count - a.count));
              return {count:cands[0].count, head:cands[0].head, region:cands[0].region};
            }
            """, g.locator.model_dump())
            n = int(data.get("count") or 0)
            ok = n >= int(g.min_items)
            return GoalResult(status=GoalStatus.MET if ok else GoalStatus.UNMET,
                              score=1.0 if ok else min(0.95, n/max(1,g.min_items)),
                              reason=f"{n} items visible (need ≥{g.min_items})")
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"list_count error: {e}")

    def evaluate_filter_applied(self, g: "FilterAppliedGoal") -> "GoalResult":
        try:
            data = self.page.evaluate("""
            ({chips_any, facets_all, url_params_contains}) => {
              const norm = s => (s||'').toLowerCase();
              const vis = el => { if(!el) return false; const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
                return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none'; };
              const chipSel = '.chip,.badge,[class*="chip"],[class*="badge"],[aria-pressed="true"],[aria-selected="true"],[role="tab"][aria-selected="true"],[role="button"][aria-pressed="true"]';
              const chips = Array.from(document.querySelectorAll(chipSel)).filter(vis).map(el => (el.innerText||el.textContent||'').replace(/\s+/g,' ').trim().toLowerCase());
              const chipsOk = (chips_any||[]).map(norm).every(tok => chips.some(c => c.includes(tok)));
              const facetsSel = 'input:checked,[aria-checked="true"],[aria-selected="true"]';
              const selectedLabels = Array.from(document.querySelectorAll(facetsSel)).map(el=>{
                const id = el.id; const lab = id ? document.querySelector(`label[for="${CSS.escape(id)}"]`) : el.closest('label');
                const t = (lab && (lab.innerText||lab.textContent)||'').replace(/\s+/g,' ').trim().toLowerCase();
                return t;
              });
              const facetsOk = (facets_all||[]).map(norm).every(tok => selectedLabels.some(t => t.includes(tok)));
              const usp = new URLSearchParams(location.search);
              let paramsOk = true;
              for (const k in (url_params_contains||{})) { const want = norm(url_params_contains[k]); const got = norm(usp.get(k)); if(!got || !got.includes(want)) { paramsOk=false; break; } }
              return {chipsOk, facetsOk, paramsOk};
            }
            """, {
                "chips_any": g.chips_any,
                "facets_all": g.facets_all,
                "url_params_contains": g.url_params_contains,
            })
            ok = bool(data.get("chipsOk", True)) and bool(data.get("facetsOk", True)) and bool(data.get("paramsOk", True))
            score = 1.0 if ok else 0.0
            reason = f"chipsOk={data.get('chipsOk')} facetsOk={data.get('facetsOk')} paramsOk={data.get('paramsOk')}"
            return GoalResult(status=GoalStatus.MET if ok else GoalStatus.UNMET, score=score, reason=reason)
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"filter_applied error: {e}")

    def evaluate_field_value_set(self, g: "FieldValueSetGoal") -> "GoalResult":
        try:
            ok = self.page.evaluate("""
            ({labels, want, exact}) => {
              const norm = s => (s||'').toLowerCase();
              const vis = el => { if(!el) return false; const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
                return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none'; };
              const controls = [];
              const labNodes = Array.from(document.querySelectorAll('label,[aria-label]'));
              for(const l of labNodes){
                const t = ((l.getAttribute('aria-label') || l.innerText || l.textContent) || '').replace(/\s+/g,' ').trim().toLowerCase();
                if(!labels.some(w => t.includes(w))) continue;
                let ctrl = l.htmlFor ? document.getElementById(l.htmlFor) : l.querySelector('input,select,textarea,[contenteditable="true"]');
                if(!ctrl) continue; if(!vis(ctrl)) continue;
                controls.push(ctrl);
              }
              const wantN = norm(want);
              for(const el of controls){
                let v='';
                if(el.matches('[contenteditable="true"]')) v = (el.innerText||'').trim();
                else if(el.tagName==='SELECT') v = (el.value || (el.selectedOptions && el.selectedOptions[0] && el.selectedOptions[0].text) || '').trim();
                else v = (el.value || '').trim();
                const gotN = norm(v);
                if(exact ? gotN===wantN : gotN.includes(wantN)) return true;
              }
              return false;
            }
            """, {"labels": [t.lower() for t in g.label_any], "want": g.want_value, "exact": g.exact})
            return GoalResult(status=GoalStatus.MET if ok else GoalStatus.UNMET, score=1.0 if ok else 0.0,
                              reason="field value matched" if ok else "field value not matched")
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"field_value_set error: {e}")

    def evaluate_upload_attached(self, g: "UploadAttachedGoal") -> "GoalResult":
        try:
            ok = self.page.evaluate("""
            ({labels, wantPreview}) => {
              const norm = s => (s||'').toLowerCase();
              const vis = el => { if(!el) return false; const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
                return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none'; };
              const labNodes = Array.from(document.querySelectorAll('label,[aria-label]'));
              const files = [];
              for(const l of labNodes){
                const t = ((l.getAttribute('aria-label') || l.innerText || l.textContent) || '').replace(/\s+/g,' ').trim().toLowerCase();
                if(!labels.some(w => t.includes(w))) continue;
                let input = l.htmlFor ? document.getElementById(l.htmlFor) : l.querySelector('input[type="file"]');
                if(!input) continue; if(!vis(input)) continue;
                if(input.files && input.files.length>0) files.push(Array.from(input.files).map(f=>f.name));
                if(wantPreview){
                  const area = l.closest('section,div,form') || document.body; const txt = (area.innerText||'').toLowerCase();
                  if(/uploaded|attached|selected|resume|cv/.test(txt)) return true;
                }
              }
              return files.length>0;
            }
            """, {"labels": [t.lower() for t in g.label_any], "wantPreview": g.require_preview_label})
            return GoalResult(status=GoalStatus.MET if ok else GoalStatus.UNMET, score=1.0 if ok else 0.0,
                              reason="upload attached" if ok else "no upload detected")
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"upload_attached error: {e}")

    def evaluate_new_tab_opened(self, g: "NewTabOpenedGoal") -> "GoalResult":
        try:
            ctx = getattr(self.page, 'context', None)
            pages = ctx.pages if ctx else []
            matched = False
            for p in pages:
                if p is self.page: 
                    continue
                try:
                    url = (p.url or '').lower(); title = (p.title() or '').lower()
                except Exception:
                    url = (p.url or '').lower(); title = ''
                ok_url = (not g.url_contains) or any(tok.lower() in url for tok in g.url_contains)
                ok_title = (not g.title_contains) or any(tok.lower() in title for tok in g.title_contains)
                if ok_url and ok_title:
                    matched = True; break
            return GoalResult(status=GoalStatus.MET if matched else GoalStatus.UNMET, score=1.0 if matched else 0.0,
                              reason="new tab matched" if matched else "no matching new tab")
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"new_tab_opened error: {e}")

    def evaluate_repeat_until(self, g: "RepeatUntilGoal") -> "GoalResult":
        last = GoalResult(status=GoalStatus.UNMET, score=0.0, reason="not started")
        for _ in range(int(max(1, g.max_iters))):
            last = self.evaluate_goal(g.goal)
            if last.status == GoalStatus.MET:
                return last
            try:
                self.page.wait_for_timeout(int(max(0, g.sleep_ms)))
            except Exception:
                pass
        return last
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
    def evaluate_list_item_opened(self, g: "ListItemOpenedGoal") -> GoalResult:
        # --- Track current and previous URLs for URL-change detection ---
        print(f"[ListItemOpened] init: start={getattr(self,'_goal_start_url',None)} prev={getattr(self,'_lio_last_seen_url',None)} curr={(self.page.url or '').lower()}")
        
        curr_url = (self.page.url or "").lower()
        prev_seen_url = getattr(self, "_lio_last_seen_url", None)

        # If this is the first time we evaluate during this goal, remember where we started
        try:
            if not getattr(self, "_goal_start_url", None):
                setattr(self, "_goal_start_url", curr_url)
        except Exception:
            pass

        # Always record the last-seen URL for the next pass (even if we early-return)
        try:
            setattr(self, "_lio_last_seen_url", curr_url)
        except Exception:
            pass

        # --- Early success: URL changed (detail page) even if a list-like nav exists ---
        if getattr(g, "require_url_change", False):
            try:
                start_url_raw = getattr(self, "_goal_start_url", None)
                start_norm = (start_url_raw or "").split('#', 1)[0].lower()
                prev_norm = (prev_seen_url or "").split('#', 1)[0].lower()
                curr_norm = (curr_url or "").split('#', 1)[0].lower()

                changed_vs_prev = bool(prev_norm) and (prev_norm != curr_norm)
                changed_vs_start = bool(start_norm) and (start_norm != curr_norm)
                # Same-origin guard
                same_origin_ok = True
                if start_norm:
                    from urllib.parse import urlparse
                    try:
                        s, c = urlparse(start_norm), urlparse(curr_norm)
                        same_origin_ok = (s.scheme, s.netloc) == (c.scheme, c.netloc)
                    except Exception:
                        same_origin_ok = True
                print(f"[ListItemOpened] early-check: start_raw={start_url_raw} prev_raw={prev_seen_url} curr_raw={curr_url} | start={start_norm} prev={prev_norm} curr={curr_norm} changed_vs_prev={changed_vs_prev} changed_vs_start={changed_vs_start} same_origin_ok={same_origin_ok}")
                if (changed_vs_prev or changed_vs_start) and same_origin_ok:
                    print("[ListItemOpened] early-check: URL changed -> MET (skip scanning)")
                    return GoalResult(status=GoalStatus.MET, score=1.0, reason="navigated to a new page (early URL-change)")
            except Exception:
                pass
        
        try:
            data = self.page.evaluate("""
            (locator, item) => {
            const norm = s => (s||'').toLowerCase();
            const txt = el => (el && (el.innerText || el.textContent) || '').replace(/\\s+/g,' ').trim();
            const vis = el => {
                if (!el) return false;
                const r = el.getBoundingClientRect();
                const s = getComputedStyle(el);
                return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none';
            };
            const headingAbove = el => {
                let n = el; const top = el.getBoundingClientRect().top; let best='';
                while (n && n!==document.body) {
                const prev = n.previousElementSibling;
                if (!prev) { n = n.parentElement; continue; }
                const heads = prev.querySelectorAll('h1,h2,h3,h4'); 
                for (const h of heads) {
                    if (!vis(h)) continue;
                    const d = top - h.getBoundingClientRect().bottom;
                    if (d>=0 && d<=250) { const t=txt(h); if (t && t.length>best.length) best=t; }
                }
                n = prev;
                }
                return best;
            };
            const sectionOf = el => (
                (el.closest('header') && 'header') || (el.closest('footer') && 'footer') ||
                (el.closest('nav,[role="navigation"]') && 'navigation') ||
                (el.closest('aside,[role="complementary"]') && 'sidebar') ||
                (el.closest('dialog,[role="dialog"],[role="alertdialog"]') && 'modal') ||
                (el.closest('main,[role="main"]') && 'content') || 'content'
            );

            const containers = new Set();
            document.querySelectorAll('ul,ol,[role="list"],[role="feed"]').forEach(c=>containers.add(c));
            document.querySelectorAll('table,[role="grid"],[role="rowgroup"]').forEach(c=>containers.add(c));

            // big repeated wrappers fallback
            document.querySelectorAll('main,[role="main"],body').forEach(root=>{
                const kids = Array.from(root.querySelectorAll('*')).filter(e=>e.children && e.children.length>3);
                for (const k of kids) {
                const ch = Array.from(k.children).filter(e=>vis(e));
                const withLinks = ch.filter(c=>c.querySelector('a[href]'));
                if (withLinks.length >= 6) containers.add(k);
                }
            });

            const kinds = (locator && locator.container_kind) || 'auto';
            const minN = (locator && locator.min_items) || 3;
            const wantRegion = locator && locator.region && locator.region.toLowerCase();
            const wantHead   = (locator && (locator.heading_contains||[])).map(norm);
            const wantAria   = (locator && (locator.aria_label_contains||[])).map(norm);

            const cands = [];
            containers.forEach(c=>{
                if (!vis(c)) return;
                if (kinds!=='auto') {
                if (kinds==='ul' && c.tagName!=='UL') return;
                if (kinds==='ol' && c.tagName!=='OL') return;
                if (kinds==='table' && c.tagName!=='TABLE') return;
                if (kinds==='grid' && !c.matches('table,[role="grid"]')) return;
                if (kinds==='feed' && !c.matches('[role="feed"],[role="list"]')) return;
                }
                const region = sectionOf(c);
                const head = headingAbove(c);
                const aria = norm(c.getAttribute('aria-label') || '');
                let items = [];
                const rows = c.tagName==='TABLE' ? Array.from(c.querySelectorAll('tr')) : Array.from(c.children);
                for (const r of rows) {
                if (!vis(r)) continue;
                const a = r.querySelector('a[href]') || r.closest('a[href]');
                const label = a ? txt(a) : txt(r);
                if (!label) continue;
                items.push({
                    label, href: a ? a.href : null,
                    domain: a ? new URL(a.href, location.href).hostname.toLowerCase() : '',
                    visible: vis(r),
                    y: r.getBoundingClientRect().top,
                    docY: r.getBoundingClientRect().top + (window.scrollY || document.documentElement.scrollTop || 0)
                });
                }
                if (items.length < minN) return;
                const score = 
                (wantHead.length && wantHead.some(h=>norm(head).includes(h)) ? 2 : 0) +
                (wantAria.length && wantAria.some(h=>aria.includes(h)) ? 1 : 0) +
                (wantRegion && wantRegion===region ? 1 : 0) +
                Math.min(3, Math.floor(items.length/10));
                cands.push({container:c, items, head, region, score});
            });

            if (!cands.length) return {found:null, pool:[], reason:'no list-like containers'};

            cands.sort((a,b)=>b.score-a.score);
            const best = cands[0];

            // If index-based requested, we return that directly (we'll finalize in Python)
            const indexMode = (item && (item.nth!=null || item.position==='last'));
            return {found:null, pool:best.items.slice(0,200), indexMode, head:best.head, region:best.region};
            }
            """, g.locator.model_dump(), g.item.model_dump())
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"list scan failed: {e}")

        # --- Scan summary print ---
        try:
            pool0 = data.get("pool") or []
            preview = [{"label": (it.get("label") or "")[:60], "href": it.get("href")} for it in pool0[:3]]
            print(f"[ListItemOpened] scan: indexMode={data.get('indexMode')} head={data.get('head')} region={data.get('region')} pool_len={len(pool0)} preview={preview}")
        except Exception:
            pass

        pool = data.get("pool") or []
        if not pool:
            # Fallbacks when list container isn't present anymore (likely navigated to detail)
            if getattr(g, "require_url_change", False):
                try:
                    print(f"[ListItemOpened] empty-pool: start={getattr(self,'_goal_start_url',None)} prev={prev_seen_url} curr={curr_url} last_href={getattr(self,'_last_listitem_href',None)}")
                except Exception:
                    pass
                # 1) Prefer exact match against previously chosen href, if we captured it
                try:
                    prev_href = getattr(self, "_last_listitem_href", None)
                    if prev_href:
                        met = prev_href.split("#")[0] == curr_url.split("#")[0]
                        try:
                            print(f"[ListItemOpened] empty-pool href-match: prev_href={prev_href} curr={curr_url} -> MET={met}")
                        except Exception:
                            pass
                        if met:
                            return GoalResult(status=GoalStatus.MET, score=1.0, reason="navigated to target item (href match)")
                except Exception:
                    pass
                # 2) Otherwise, accept a clear URL change compared to our last seen (or start) URL
                try:
                    start_url_raw = getattr(self, "_goal_start_url", None)
                    start_norm = (start_url_raw or "").split('#', 1)[0].lower()
                    prev_norm = (prev_seen_url or "").split('#', 1)[0].lower()
                    curr_norm = (curr_url or "").split('#', 1)[0].lower()
                    # URL changed during this goal attempt?
                    changed_vs_prev = bool(prev_norm) and (prev_norm != curr_norm)
                    changed_vs_start = bool(start_norm) and (start_norm != curr_norm)
                    # Same-origin guard: avoid counting new-tab or cross-origin redirects as success unless required
                    same_origin_ok = True
                    if start_norm:
                        from urllib.parse import urlparse
                        try:
                            s, c = urlparse(start_norm), urlparse(curr_norm)
                            same_origin_ok = (s.scheme, s.netloc) == (c.scheme, c.netloc)
                        except Exception:
                            same_origin_ok = True
                    try:
                        print(f"[ListItemOpened] empty-pool change-check: start_raw={start_url_raw} prev_raw={prev_seen_url} curr_raw={curr_url} | start={start_norm} prev={prev_norm} curr={curr_norm} changed_vs_prev={changed_vs_prev} changed_vs_start={changed_vs_start} same_origin_ok={same_origin_ok}")
                    except Exception:
                        pass
                    if (changed_vs_prev or changed_vs_start) and same_origin_ok:
                        try:
                            print("[ListItemOpened] empty-pool change-check: URL changed -> MET")
                        except Exception:
                            pass
                        return GoalResult(status=GoalStatus.MET, score=1.0, reason="navigated to a new page (URL changed)")
                except Exception:
                    pass
            try:
                print("[ListItemOpened] empty-pool: UNMET (no items & no URL change)")
            except Exception:
                pass
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason="no items in chosen list")
        # Always persist last-seen URL for next evaluation pass
        try:
            setattr(self, "_lio_last_seen_url", curr_url)
        except Exception:
            pass

        # --- Index mode? (kept behavior) ---
        it = g.item
        if it.nth is not None or it.position == "last":
            idx = (max(0, min(len(pool)-1, (it.nth or 1)-1)) if it.nth is not None else len(pool)-1)
            chosen = pool[idx]
        else:
            # --- Content scoring ---
            import re as _re
            def _score(p):
                s = 0
                label = (p.get("label") or "").lower()
                href  = (p.get("href") or "").lower()
                dom   = (p.get("domain") or "").lower()

                # visibility / link constraints
                if it.require_visible and not p.get("visible"): return -1e9
                if it.require_link and not href: return -1e9

                # excludes
                if any(x.lower() in label for x in (it.exclude_text or [])): return -1e9

                # text_any
                if it.text_any:
                    if any(t.lower() in label for t in it.text_any): s += 3
                    else: s -= 1  # soft penalty if absent

                # text_all
                if it.text_all:
                    if all(t.lower() in label for t in it.text_all): s += 5
                    else: return -1e9  # hard fail

                # regex
                if it.regex:
                    try:
                        if _re.search(it.regex, label, _re.I): s += 5
                        else: return -1e9
                    except Exception:
                        pass

                # url/domain filters
                if it.url_contains and any(u.lower() in href for u in it.url_contains): s += 3
                if it.domain_in and dom in [d.lower() for d in it.domain_in]: s += 3

                # position bias: items higher on the page get a tiny boost
                try:
                    y = float(p.get("y") or 0.0)
                    s += max(0.0, 1.0 - min(1.0, y/2000.0))
                except Exception:
                    pass
                return s

            scored = [(p, _score(p)) for p in pool]
            # filter out hard fails
            scored = [(p, sc) for (p, sc) in scored if sc>-1e9]
            if not scored:
                return GoalResult(status=GoalStatus.UNMET, score=0.0, reason="no item matches content filters")

            # Optional semantic re-rank (small, controlled use)
            chosen = None
            if it.semantic_query:
                # Preselect top-K by cheap score, then ask the LLM which label best matches the query.
                scored.sort(key=lambda x: x[1], reverse=True)
                top = scored[: max(3, min(it.top_k_semantic, len(scored)))]
                labels = [p["label"] for p,_ in top]
                try:
                    pick = self.generate_text(
                        system_prompt=(
                            "Pick the SINGLE best label that satisfies the user's intent. "
                            "Return EXACTLY the label text; no extra words."
                        ),
                        prompt=f"Intent: {it.semantic_query}\nChoices:\n" + "\n".join(f"- {t}" for t in labels),
                        model="gpt-5-nano",
                        reasoning_level="low",
                    )
                    pick = str(pick).strip()
                    for p,_ in top:
                        if p["label"] == pick:
                            chosen = p; break
                except Exception:
                    chosen = None

            if not chosen:
                # Fallback tie-breaking
                if it.tie_breaker == "first":
                    chosen = max(scored, key=lambda x: x[1])[0]  # still pick highest score; first if equal
                elif it.tie_breaker == "last":
                    chosen = min(scored, key=lambda x: x[1])[0]
                else:
                    chosen = max(scored, key=lambda x: x[1])[0]

        # --- Chosen item summary print ---
        try:
            print(f"[ListItemOpened] chosen: label={(chosen.get('label') or '')[:80]} href={chosen.get('href')} y={chosen.get('y')} docY={chosen.get('docY')}")
        except Exception:
            pass

        print(f"[ListItemOpened] start={getattr(self,'_goal_start_url',None)} prev={prev_seen_url} curr={curr_url}")
        # --- Persist chosen item for fallback completion detection ---
        # Persist likely target so that subsequent evaluations (after list disappears) can verify success
        try:
            setattr(self, "_last_listitem_href", (chosen.get("href") or "").lower() or None)
            setattr(self, "_last_listitem_label", chosen.get("label") or None)
        except Exception:
            pass

        # --- Completion check ---
        href = (chosen.get("href") or "").lower()
        if g.require_url_change:
            url = (self.page.url or "").lower()
            met = href and (href.split("#")[0] == url.split("#")[0])
            try:
                print(f"[ListItemOpened] check-require-url-change: href={href} url={url} met={met}")
            except Exception:
                pass
            try:
                setattr(self, "_lio_last_seen_url", curr_url)
            except Exception:
                pass
            return GoalResult(status=GoalStatus.MET if met else GoalStatus.UNMET,
                            score=1.0 if met else 0.0,
                            reason="navigated to target item" if met else "not at target yet")
        else:
            # Focus/selection variant
            try:
                docY = float(chosen.get("docY") or 0.0)
                scrolled_near = bool(self.page.evaluate(
                    "(absY) => Math.abs(((window.scrollY || document.documentElement.scrollTop || 0)) - absY) < 40",
                    docY
                ))
            except Exception:
                scrolled_near = False
            try:
                print(f"[ListItemOpened] check-focus: docY={docY} scrolled_near={scrolled_near}")
            except Exception:
                pass
            try:
                setattr(self, "_lio_last_seen_url", curr_url)
            except Exception:
                pass
            return GoalResult(status=GoalStatus.MET if scrolled_near else GoalStatus.UNMET,
                            score=1.0 if scrolled_near else 0.0,
                            reason="target item focused" if scrolled_near else "target item not focused")
            
    def _stopwhen_is_empty(self, sw: StopWhen) -> bool:
        try:
            return not any([
                bool(getattr(sw, 'url_contains', None)),
                bool(getattr(sw, 'title_contains', None)),
                bool(getattr(sw, 'headings', None)),
                bool(getattr(sw, 'body_contains', None)),
                bool(getattr(sw, 'hostname_in', None)),
                bool(getattr(sw, 'path_contains', None)),
                bool(getattr(sw, 'aria_active_tab', None)),
                bool(getattr(sw, 'zero_required_fields_remaining', False)),
            ])
        except Exception:
            return False
        
    def evaluate_page_reached(self, g: PageReachedGoal) -> GoalResult:
        sw = g.stop_when
        if hasattr(self, "_stopwhen_is_empty"):
            if self._stopwhen_is_empty(sw):
                try:
                    print("[PageReached Debug] Empty StopWhen received from inference; using change-detector fallback (require_url_change/min_strong_signals).")
                except Exception:
                    pass
        met = self._goal_stop_check(sw)
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

    def _form_status_js(self):
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
            status = self._collect_form_status_across_contexts()
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

    def _detect_prominent_surface_js(self):
        return """
        () => {
          const vis = el => { if(!el) return false; const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
            return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none'; };
          const area = el => { const r=el.getBoundingClientRect(); return Math.max(0, r.width*r.height); };
          const z = el => parseFloat(getComputedStyle(el).zIndex||'0')||0;
          const vw = (innerWidth||document.documentElement.clientWidth), vh=(innerHeight||document.documentElement.clientHeight);

          const cands = new Set();
          document.querySelectorAll('dialog,[role="dialog"],[role="alertdialog"],[aria-modal="true"]').forEach(e=>cands.add(e));
          document.querySelectorAll('div,section,aside').forEach(e=>{
            const s=getComputedStyle(e);
            if(s.position==='fixed' && vis(e)){
              const r=e.getBoundingClientRect();
              const edge = Math.min(r.top, vh - r.bottom, r.left, vw - r.right);
              const tallEnough = r.height >= Math.max(48, 0.06 * vh);
              const wideEnough = r.width >= 0.5 * vw;
              const nearEdge = edge <= 8; // pinned to an edge
              const covers = (wideEnough && tallEnough && nearEdge) || (r.width >= 0.4 * vw && r.height >= 0.12 * vh);
              if(covers) cands.add(e);
            }
          });

          let best=null, bestScore=-1;
          cands.forEach(e=>{
            if(!vis(e)) return;
            const r=e.getBoundingClientRect();
            const score = (area(e)/(vw*vh)) * 3 + z(e)*0.001 + (e.hasAttribute('open')?0.2:0) + (r.top<=8||vh-r.bottom<=8?0.3:0);
            if(score>bestScore){ best=e; bestScore=score; }
          });
          if(!best) return null;

          const buttons = Array.from(best.querySelectorAll('button,[role="button"],a[href],input[type="submit"],input[type="button"]'))
            .filter(vis)
            .map(b=>({ text:(b.innerText||b.value||'').replace(/\s+/g,' ').trim(),
                      rect: b.getBoundingClientRect() }));
          const br = best.getBoundingClientRect();
          return { kind: best.tagName.toLowerCase(), role: best.getAttribute('role')||'',
                   rect: {top:br.top,left:br.left,bottom:br.bottom,right:br.right,width:br.width,height:br.height},
                   z: z(best), area: area(best), buttons };
        }
        """;

    def _list_prominent_surfaces_js(self):
        return """
        () => {
          const vw=(innerWidth||document.documentElement.clientWidth), vh=(innerHeight||document.documentElement.clientHeight);
          const vis = el => { if(!el) return false; const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
            return r.width>0 && r.height>0 && s.visibility!=='hidden' && s.display!=='none'; };
          const area = el => { const r=el.getBoundingClientRect(); return Math.max(0, r.width*r.height); };
          const z = el => parseFloat(getComputedStyle(el).zIndex||'0')||0;
          const text = el => (el.innerText||el.textContent||'').replace(/\\s+/g,' ').trim().toLowerCase();
          const tokenize = s => Array.from(new Set(s.split(/[^a-z0-9]+/i).filter(t=>t && t.length>2))).slice(0,40);
          const nearEdge = r => Math.min(r.top, vh - r.bottom, r.left, vw - r.right) <= 8;
          const isBannerShape = r => (r.width >= 0.5*vw && r.height >= Math.max(48, 0.06*vh));
          const isBanner = el => { const r=el.getBoundingClientRect(); const s=getComputedStyle(el);
            const pos = s.position; const nearTop = r.top<=8, nearBottom = (vh - r.bottom)<=8;
            if(pos==='fixed' || pos==='sticky') return (nearTop||nearBottom) && isBannerShape(r);
            if(pos==='absolute') return isBannerShape(r) && nearEdge(r) && z(el) >= 10; // large overlay near edge
            return false;
          };
          const isModal = el => !!(el.closest('dialog') || el.getAttribute('role')==='dialog' || el.getAttribute('role')==='alertdialog' || el.getAttribute('aria-modal')==='true');

          const looksLikeCookie = el => {
            const cls = (el.className||'').toString().toLowerCase();
            const id = (el.id||'').toLowerCase();
            const t = text(el);
            return /(cookie|consent|gdpr|privacy|preferences)/.test(cls+" "+id+" "+t);
          };

          const cands=new Set();
          // dialogs
          document.querySelectorAll('dialog,[role="dialog"],[role="alertdialog"],[aria-modal="true"]').forEach(e=>{ if(vis(e)) cands.add(e); });
          // obvious overlays
          document.querySelectorAll('div,section,aside').forEach(e=>{ const s=getComputedStyle(e); if(!vis(e)) return; if(s.position==='fixed'||s.position==='sticky'||s.position==='absolute'){ if(isBanner(e) || looksLikeCookie(e)) cands.add(e); } });

          const out=[];
          cands.forEach(e=>{
            if(!vis(e)) return;
            const r=e.getBoundingClientRect();
            const btns = Array.from(e.querySelectorAll('button,[role="button"],a[href],input[type=button],input[type=submit]')).filter(vis)
                           .map(b=> (b.innerText||b.value||'').replace(/\\s+/g,' ').trim().toLowerCase()).slice(0,12);
            out.push({
              rect:{top:r.top,left:r.left,bottom:r.bottom,right:r.right,width:r.width,height:r.height},
              area_ratio: area(e)/(vw*vh),
              z: z(e),
              is_modal: isModal(e),
              is_banner: isBanner(e),
              text_tokens: tokenize(text(e)).slice(0,40),
              button_texts: Array.from(new Set(btns)),
            });
          });

          // Body-level lock (modal backdrops often disable scroll)
          const bodyLocked = (()=>{ 
            const b=getComputedStyle(document.body);
            const h=getComputedStyle(document.documentElement);
            const anyHidden = [b.overflow,b.overflowY].some(v => (v||'').includes('hidden'));
            const anyHiddenHtml = [h.overflow,h.overflowY].some(v => (v||'').includes('hidden'));
            const classFlags = (document.body.className + ' ' + document.documentElement.className).toLowerCase();
            return anyHidden || anyHiddenHtml || /modal-open|no-scroll|scroll[- ]?lock/.test(classFlags);
          })();
          return {surfaces: out, bodyLocked};
        }
        """;

    def _iou(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        ax1, ay1, ax2, ay2 = a.get('left',0.0), a.get('top',0.0), a.get('right',0.0), a.get('bottom',0.0)
        bx1, by1, bx2, by2 = b.get('left',0.0), b.get('top',0.0), b.get('right',0.0), b.get('bottom',0.0)
        inter_w = max(0.0, min(ax2,bx2) - max(ax1,bx1))
        inter_h = max(0.0, min(ay2,by2) - max(ay1,by1))
        inter   = inter_w * inter_h
        area_a  = max(0.0, (ax2-ax1) * (ay2-ay1))
        area_b  = max(0.0, (bx2-bx1) * (by2-by1))
        denom = area_a + area_b - inter
        return (inter/denom) if denom>0 else 0.0

    def _tok_sim(self, a: List[str], b: List[str]) -> float:
        if not a and not b: return 1.0
        sa, sb = set(a or []), set(b or [])
        if not sa or not sb: return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter/union if union else 0.0

    def evaluate_surface_gone(self, g: "SurfaceGoneGoal") -> "GoalResult":
        try:
            data = self.page.evaluate(self._list_prominent_surfaces_js())
        except Exception as e:
            return GoalResult(status=GoalStatus.UNMET, score=0.0, reason=f"surface list error: {e}")

        want = g.signature.model_dump()

        def _is_placeholder(sig: Dict[str, Any]) -> bool:
            try:
                r = sig.get("rect", {}) or {}
                return (float(sig.get("area_ratio", 0.0)) <= 0.0) or \
                    (float(r.get("width", 0.0)) <= 0.0) or \
                    (float(r.get("height", 0.0)) <= 0.0)
            except Exception:
                return True

        if _is_placeholder(want):
            snap = getattr(self, "_last_surface_signature", None)
            if isinstance(snap, dict) and not _is_placeholder(snap):
                want = snap
            else:
                return GoalResult(status=GoalStatus.UNMET, score=0.0, reason="no prior surface signature to compare")

        surfaces = data.get('surfaces') or []
        body_locked = bool(data.get('bodyLocked'))

        # Helper
        def _area(r: Dict[str, float]) -> float:
            try:
                return max(0.0, (float(r.get('right',0))-float(r.get('left',0))) * (float(r.get('bottom',0))-float(r.get('top',0))))
            except Exception:
                return 0.0

        want_area = _area(want.get('rect', {}))
        want_is_modal = bool(want.get('is_modal'))
        want_is_banner = bool(want.get('is_banner'))

        # Adaptive thresholds
        iou_thr = 0.35 if want_is_modal else (0.30 if want_is_banner else 0.40)
        sim_thr = 0.45 if (want_is_modal or want_is_banner) else 0.50

        # Score candidates and decide if any sufficiently match
        scored = []
        best_iou, best_sim, still_present = 0.0, 0.0, False
        for s in surfaces:
            # type consistency: if signature says banner/modal, prefer same type
            if want_is_modal and not s.get('is_modal'):  # require modal when we expect modal
                pass  # keep, but penalize via score
            if want_is_banner and not s.get('is_banner'):
                pass

            iou = self._iou(s.get('rect',{}), want.get('rect',{}))
            sim = max(
                self._tok_sim(s.get('text_tokens',[]), want.get('text_tokens',[])),
                self._tok_sim(s.get('button_texts',[]), want.get('button_texts',[])),
            )
            if iou>best_iou: best_iou=iou
            if sim>best_sim: best_sim=sim

            # penalize type mismatch
            type_penalty = 0.15 if ((want_is_modal and not s.get('is_modal')) or (want_is_banner and not s.get('is_banner'))) else 0.0
            combined = max(0.0, 0.6*iou + 0.4*sim - type_penalty)

            scored.append({
                'iou': iou,
                'sim': sim,
                'combined': combined,
                'is_modal': bool(s.get('is_modal')),
                'is_banner': bool(s.get('is_banner')),
                'area_ratio': float(s.get('area_ratio') or 0.0),
                'rect': s.get('rect', {}),
            })

        # Determine presence: either strong combined score, or (iou & sim) both above adaptive thresholds
        for cand in scored:
            if (cand['combined'] >= 0.50) or (cand['iou'] >= iou_thr and cand['sim'] >= sim_thr):
                still_present = True
                break

        ok_surface = not still_present
        ok_unlock = (not g.require_scroll_unlock) or (not body_locked)
        ok = ok_surface and ok_unlock

        score = 1.0 if ok else (0.0 if still_present else (0.6 if not ok_unlock else 0.0))
        reason = f"surfaceGone={ok_surface}, scrollUnlock={ok_unlock}, bestIoU={best_iou:.2f}, bestSim={best_sim:.2f}, bodyLocked={body_locked}"

        # Debug: print top 3 by combined score
        try:
            top3 = sorted(scored, key=lambda d: d['combined'], reverse=True)[:3]
            print("[SurfaceGone Debug] want:", {k: want.get(k) for k in ('is_modal','is_banner','rect','text_tokens','button_texts')})
            print("[SurfaceGone Debug] candidates:")
            for i, c in enumerate(top3):
                print(f"  #{i+1} combined={c['combined']:.2f} iou={c['iou']:.2f} sim={c['sim']:.2f} modal={c['is_modal']} banner={c['is_banner']} area_ratio={c['area_ratio']:.3f}")
            print("[SurfaceGone Debug] ok_surface:", ok_surface, "ok_unlock:", ok_unlock)
        except Exception:
            pass

        try:
            setattr(self, "_last_surface_signature", None)
        except Exception:
            pass

        return GoalResult(status=GoalStatus.MET if ok else GoalStatus.UNMET, score=score, reason=reason)

    def detect_prominent_surface(self) -> Optional[Dict[str, Any]]:
        try:
            return self.page.evaluate(self._detect_prominent_surface_js())
        except Exception:
            return None

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
        if t == GoalKind.LIST_ITEM_OPENED:
            return self.evaluate_list_item_opened(goal)      # type: ignore
        if t == GoalKind.ELEMENT_VISIBLE:
            return self.evaluate_element_visible(goal)       # type: ignore
        if t == GoalKind.ELEMENT_GONE:
            return self.evaluate_element_gone(goal)          # type: ignore
        if t == GoalKind.LIST_COUNT:
            return self.evaluate_list_count(goal)            # type: ignore
        if t == GoalKind.FILTER_APPLIED:
            return self.evaluate_filter_applied(goal)        # type: ignore
        if t == GoalKind.FIELD_VALUE_SET:
            return self.evaluate_field_value_set(goal)       # type: ignore
        if t == GoalKind.UPLOAD_ATTACHED:
            return self.evaluate_upload_attached(goal)       # type: ignore
        if t == GoalKind.NEW_TAB_OPENED:
            return self.evaluate_new_tab_opened(goal)        # type: ignore
        if t == GoalKind.SURFACE_GONE:
            return self.evaluate_surface_gone(goal)         # type: ignore
        if t == GoalKind.REPEAT_UNTIL:
            return self.evaluate_repeat_until(goal)          # type: ignore
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
    
    def _pick_best_surface(self, scan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Heuristic: choose the most prominent surface by area_ratio, then z-index."""
        try:
            if not scan:
                return None
            surfaces = scan.get("surfaces") if isinstance(scan, dict) else None
            if not surfaces:
                return None
            return sorted(
                surfaces,
                key=lambda s: (float(s.get("area_ratio", 0.0)), float(s.get("z", 0.0))),
                reverse=True
            )[0]
        except Exception:
            return None

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
    def _looks_like_first_list_item(self, low: str) -> bool:
        return any(p in low for p in [
            "first item", "top item", "first link", "top link",
            "first result", "top result", "top story", "first story"
        ])

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
    ) -> RunSpec:
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
        wants_first_item = self._looks_like_first_list_item(low)
        
        # NEW: list-item intent detection (index or content)
        def _extract_quoted_terms(s: str) -> List[str]:
            return [m.strip() for m in re.findall(r"[\"“”'‘’]([^\"“”'‘’]{2,80})[\"“”'‘’]", s or "") if m.strip()]
        quoted = _extract_quoted_terms(subprompt)

        # --- NEW: map common phrasings → new goals ---
        # ElementVisibleGoal: "show/see/find <'text'> [button/link/heading]"
        role_any: List[str] = []
        if any(tok in low for tok in [" button", "button "]):
            role_any.append("button")
        if any(tok in low for tok in [" link", "link "]):
            role_any.append("link")
        if any(tok in low for tok in [" heading", "title", "header"]):
            role_any.append("heading")
        if (any(k in low for k in ["show", "see", "visible", "find"]) and quoted):
            goals.append(ElementVisibleGoal(
                match=ElementMatch(text_any=quoted, role_any=role_any or []),
                min_count=1,
            ))

        # Overlay dismiss intent → prefer SurfaceGoneGoal with runtime-bound signature, fallback to ElementGoneGoal
        if any(k in low for k in ["close", "dismiss", "hide", "remove"]) and \
           any(k in low for k in ["modal", "dialog", "banner", "cookie", "toast", "popup", "pop-up", "consent", "overlay"]):
            # Broad terms so it works across sites; if user quoted specifics, include those too
            gone_terms = [t for t in ["modal", "dialog", "banner", "cookie", "toast", "popup", "overlay", "consent"] if t in low]
            if not gone_terms and quoted:
                gone_terms = quoted

            # Placeholder signature; planner/executor should bind real signature from DOM (see detect_prominent_surface/_list_prominent_surfaces_js)
            placeholder_sig = SurfaceSignature(
                rect={"top":0.0, "left":0.0, "bottom":0.0, "right":0.0, "width":0.0, "height":0.0},
                area_ratio=0.0, z=0.0, is_modal=False, is_banner=False,
                text_tokens=[], button_texts=[]
            )

            goals.append(AnyOfGoal(goals=[
                SurfaceGoneGoal(signature=placeholder_sig, require_scroll_unlock=True),
                ElementGoneGoal(match=ElementMatch(text_any=gone_terms))
            ]))

        # ListCountGoal: "load/show at least N results/items" (works well with infinite scroll)
        min_items = None
        m_at_least = re.search(r"(at least|min(?:imum)?)\s+(\d{1,4})", low)
        m_plain_n = re.search(r"\b(\d{1,4})\b(?=\s+(?:results|items|rows|jobs))", low)
        if m_at_least:
            min_items = int(m_at_least.group(2))
        elif m_plain_n:
            min_items = int(m_plain_n.group(1))
        if min_items and min_items > 0:
            goals.append(ListCountGoal(locator=ListLocator(region=SectionIn.CONTENT.value, container_kind="auto"), min_items=min_items))

        # FilterAppliedGoal: "filter by 'X' and 'Y'" / "only show 'Remote' + 'iOS'"
        if ("filter by" in low or "only show" in low or "include" in low) and quoted:
            goals.append(FilterAppliedGoal(chips_any=[q.lower() for q in quoted], facets_all=[q.lower() for q in quoted]))

        # FieldValueSetGoal: "set/type/enter/fill '<label>' to/with '<value>'"
        m_set = re.search(r"(?:set|type|enter|fill)\s*[\"'“”‘’]?(.+?)[\"'“”‘’]?\s*(?:to|with|=)\s*[\"'“”‘’]?(.+?)[\"'“”‘’]?$", s, re.I)
        if m_set:
            label = m_set.group(1).strip()
            val = m_set.group(2).strip()
            if label and val:
                goals.append(FieldValueSetGoal(label_any=[label], want_value=val, exact=True))

        # UploadAttachedGoal: "attach/upload/choose file/select file ... resume/cv/cover letter"
        if any(k in low for k in ["attach", "upload", "choose file", "select file"]) and \
           any(k in low for k in ["resume", "cv", "cover letter", "attachment", "file"]):
            goals.append(UploadAttachedGoal())

        # NewTabOpenedGoal: "open in a new tab"
        if "new tab" in low:
            goals.append(NewTabOpenedGoal(title_contains=[q.lower() for q in quoted] if quoted else []))

        # RepeatUntilGoal: "until we have N results/items" → wrap the most recent list goal
        m_until = re.search(r"until\s+(?:we\s+)?(?:have\s+)?(\d{1,4})\s+(?:results|items|rows|jobs)", low)
        if m_until and goals:
            try:
                inner = goals[-1]
                goals[-1] = RepeatUntilGoal(goal=inner, max_iters=15, sleep_ms=300)
            except Exception:
                pass
        
        if wants_first_item:
            # Minimal locator; you can enrich with section/heading from page synthesis if you’d like
            list_goal = ListItemOpenedGoal(
                locator=ListLocator(region=SectionIn.CONTENT.value, container_kind="auto", min_items=3),
                item=ItemSpec(position="first", text_any=quoted, exclude_text=["job","sponsored","promotion"], require_link=True, require_visible=True),
                require_url_change=True
            )
            goals = [list_goal]  # single clear goal

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
                return RunSpec(
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

        return RunSpec(
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
    
    def _merge_preferences(self, base: Dict[str, Any], hints: RunSpec) -> Dict[str, Any]:
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


    def _active_adapter_for_url(self, url: str, hints: RunSpec) -> Optional[Adapter]:
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



    def _snapshot_from_hints(self, h: RunSpec, active: Optional[Adapter]) -> str:
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


    def _build_hard_constraints(self, meta_instructions: str, hints: RunSpec, active: Optional[Adapter]) -> str:
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
    hints: RunSpec | Dict[str, Any] | None = None,   # NEW
):
        """
        Executes a compressed phrase using ONE-PASS detection+selection+planning per viewport.
        - NEW: accepts `hints` (RunSpec) to specify goals/stop conditions, profile values, assets, policies, adapters, synonyms.
        - Merges hints into HARD CONSTRAINTS and into preferences for value sourcing.
        - Continues to reset state on navigation and enforces constraints with verifier + local filters.
        """
        from pydantic import ValidationError
        
        # Capture start URL for this run if not already set
        try:
            if not getattr(self, "_goal_start_url", None):
                setattr(self, "_goal_start_url", (self.page.url or "").lower())
        except Exception:
            pass

        # Normalize hints
        if hints is None:
            hints = RunSpec()
        elif isinstance(hints, dict):
            try:
                hints = RunSpec.model_validate(hints)
            except ValidationError as e:
                print(f"[hints] validation failed, proceeding with defaults: {e}")
                hints = RunSpec()

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

        def _unified_system_prompt(user_prompt: str,
                                        viewport_meta: dict,
                                        goal: Goal | None,
                                        uxhints: RunSpec | None) -> str:
            goal_json = goal.model_dump() if goal else {}
            policy_json = (uxhints.policies.model_dump() if (uxhints and uxhints.policies) else {})
            stopwhen_json = {}
            # Try to extract a StopWhen if present inside a PageReachedGoal or similar
            if goal and getattr(goal, "type", None) == "page_reached":
                stopwhen_json = getattr(goal, "stop_when", {}).model_dump() if hasattr(goal, "stop_when") else {}

            return f"""
                        You are a vision model that must detect UI targets and produce a precise action plan.
                        Use ONLY the viewport screenshot and the GOAL/POLICIES below to decide *what to look for* and *where*.

                        ### INPUT INTENT
                        - user_prompt: {user_prompt}

                        ### GOAL (structured, generic; do not assume any site)
                        {goal_json}

                        ### POLICIES (safety/biases; do not violate)
                        {policy_json}

                        ### NAVIGATION HINTS (optional)
                        {stopwhen_json}

                        ### VIEWPORT META
                        {viewport_meta}

                        ### DETECTION PRINCIPLES (apply in this order)
                        1) Determine the **prominent surface** to search:
                        - If a visible dialog/modal/sheet/overlay exists, search **inside it only**; ignore greyed/behind content.
                        - Else search the main **content** region; avoid header/footer/navigation unless GOAL suggests otherwise.
                        - Prefer the surface with the highest z-index or strongest visual prominence.

                        2) Generate **candidates** and score them using a consistent rubric:
                        - +Text relevance: includes any GOAL text hints (exact/near, case-insensitive).
                        - +Role/affordance: matches desired role (button/link/heading/input), looks interactive, enabled & visible.
                        - +Visibility: non-zero size, on top (not occluded), readable contrast.
                        - +Spatial cues: near relevant headings/labels; in requested region (header/footer/sidebar/modal/content).
                        - +List goals: container looks list-like (ul/ol/table/grid/feed/repeated siblings) and item matches {{"text_any","text_all","regex","url_contains","domain_in"}}.
                        - +Navigation/Page goals: headings/title/url substrings close to StopWhen.
                        - −Penalize: disabled/behind overlay/irrelevant roles/“manage/settings”-like detours **unless** goal requires them.

                        3) **Multi-context awareness**:
                        - If candidate is inside a visible iframe, annotate `in_iframe=true`.
                        - If target is off-screen, set `needs_scroll=true` and include a `scroll` step BEFORE clicking.
                        - If inside a shadow root, annotate `shadow_host=true` with host box.

                        4) **Choose and justify**:
                        - Keep TOP 3 candidates with scores and reasons. Pick the best one for the action plan.
                        - Provide `confidence` ∈ [0,1] and short `reason`.

                        ### OUTPUT (STRICT JSON per schema you were given)
                        Return a `ModelPlan`:
                        - detection_result: list of detected items (top surface + top 3 candidates). Each item must include:
                        - description, item_type, clickable flag, box_2d (0–1000 normalized), section_in, confidence,
                        - any flags: in_dialog, is_behind_modal, in_iframe, shadow_host (if applicable)
                        - steps: a minimal sequence to achieve the GOAL (e.g., optional scroll → click/move/type/press), with target by index
                        - reason, confidence

                        Coordinate contract: click at the CENTER of the target; coordinates are CSS px in the **visible viewport**.
                        Do NOT submit forms or navigate unless the GOAL explicitly requires it.
                        If uncertain, pick the highest-scoring candidate and include strong alternates with reasons.
                        """

        def _select_primary_goal(subprompt: str, uxhints: RunSpec | None) -> Optional[Goal]:
            """Pick the single most specific goal to condition the detector with.
            Preference order:
            1) Any non-navigation goal (e.g., form/list/element/login/...)
            2) Otherwise fall back to the first goal (often PageReachedGoal)
            """
            if not uxhints:
                return None
            try:
                gs: List[Goal] = uxhints.normalized_goals()
            except Exception:
                gs = (uxhints.goals or [])
            if not gs:
                return None
            # Prefer a goal that is not a generic page_reached navigation goal
            for g in gs:
                if getattr(g, "type", None) not in ("page_reached",):
                    return g
            return gs[0]
        
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
            # cookie_handler.handle_cookies_with_dom()

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


                active_adapter = _active_adapter()
                # Merge preferences with hints (profile/assets/auth/dictionary/policies)
                hard_constraints = self._build_hard_constraints(meta_instructions, hints, active_adapter)

                # Reset last surface signature for this run and give overlays a moment to appear
                self._last_surface_signature = None
                try:
                    self.page.wait_for_timeout(50)
                except Exception:
                    pass
                
                primary_goal = _select_primary_goal(prompt, hints)
                system_prompt = _unified_system_prompt(
                    user_prompt=prompt,
                    viewport_meta=(meta.model_dump() if hasattr(meta, "model_dump") else getattr(meta, "__dict__", meta)),
                    goal=primary_goal,
                    uxhints=hints,
                )
                user_prompt = "Plan now. Return ONLY the JSON for ModelPlan.\n\nBANS:\n" + (hard_constraints or "")

                # Prime a pre-click surface signature for SURFACE_GONE verification
                try:
                    _scan = self.page.evaluate(self._list_prominent_surfaces_js())
                    self._last_surface_signature = self._pick_best_surface(_scan)
                    if not self._last_surface_signature:
                        # Retry once after a short settle (animations/late banners)
                        try:
                            self.page.wait_for_timeout(120)
                            _scan = self.page.evaluate(self._list_prominent_surfaces_js())
                            self._last_surface_signature = self._pick_best_surface(_scan)
                        except Exception:
                            pass
                except Exception:
                    self._last_surface_signature = None
                
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
        subprompt_hints: Optional[Mapping[str, Union[RunSpec, Dict[str, Any]]]] = None,
        strict: bool = False,
    ):
        """
        Execute a prompt broken into sub-prompts, OR execute an explicit list of
        sub-prompts with supplied RunSpec.

        Args:
            prompt: Optional full instruction to decompose. If omitted/blank, we use
                    the keys of `subprompt_hints` (in insertion order) as sub-prompts.
            subprompt_hints: Optional mapping {sub-prompt -> RunSpec | dict}. If a
                    hint is not provided for a sub-prompt (when `prompt` is used),
                    we'll infer one unless `strict=True`.
            strict: If True, require a manual hint for every sub-prompt (when
                    `prompt` is provided). If prompt is omitted, all sub-prompts
                    MUST come from `subprompt_hints`.
        """
        # Build manual hints map (canonicalized)
        manual_map: Dict[str, RunSpec] = {}
        if subprompt_hints:
            for k, v in subprompt_hints.items():
                if isinstance(v, dict):
                    try:
                        v = RunSpec.model_validate(v)
                    except ValidationError as e:
                        raise ValueError(f"Invalid RunSpec for '{k}': {e}") from e
                elif not isinstance(v, RunSpec):
                    raise TypeError(f"Hint for '{k}' must be RunSpec or dict, got {type(v).__name__}")
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

            # Safety: if anything snuck in as a dict, validate to RunSpec
            if isinstance(hints, dict):
                try:
                    hints = RunSpec.model_validate(hints)
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
                # Snapshot the URL *before* a potential navigation so evaluators can detect change
                try:
                    setattr(self, "_lio_last_seen_url", (self.page.url or "").lower())
                except Exception:
                    pass
                
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