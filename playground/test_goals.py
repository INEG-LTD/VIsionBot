# playground/test_goals.py
from __future__ import annotations
import os, time, multiprocessing, tempfile, textwrap, argparse
from playwright.sync_api import sync_playwright
from pathlib import Path

# Import your bot
import sys

from pydantic import BaseModel
from typing import Optional
sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root on sys.path
from vision_bot import Goal, GoalKind, ItemSpec, ListItemOpenedGoal, VisionCoordinator, RunSpec, Assets, UploadPolicy, Auth

class RunTest(BaseModel):
    url: str
    prompt: str
    hints: Optional[RunSpec] = None

# Start the Flask server inside this process (subprocess-safe)
def _run_server():
    from app import create_app  # local import so sys.path is right when spawned
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)

def _ensure_resume(tmpdir: Path) -> Path:
    """Create a tiny fake PDF so the upload test has something to attach."""
    pdf = tmpdir / "resume.pdf"
    if not pdf.exists():
        pdf.write_bytes(
            b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R>>endobj\n"
            b"4 0 obj<</Length 35>>stream\nBT /F1 12 Tf 72 100 Td (Hello Resume) Tj ET\nendstream endobj\n"
            b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000113 00000 n \n0000000207 00000 n \n"
            b"trailer <</Size 5/Root 1 0 R>>\nstartxref\n300\n%%EOF"
        )
    return pdf

from typing import Optional

def run_tests(base_url: str = "http://127.0.0.1:5001", *, only: Optional[str] = None, match: Optional[str] = None, headless: bool = False):
    # Launch browser at DPR 1.0 (your VisionCoordinator enforces this)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        page = ctx.new_page()

        # seed preferences (used by infer + planner)
        vc = VisionCoordinator(
            page,
            model="gemini-2.5-flash-lite",
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

        tmpdir = Path(tempfile.gettempdir())
        resume_path = str(_ensure_resume(tmpdir))

        tests = [
            # ELEMENT_VISIBLE
            RunTest(url=f"{base_url}/", prompt="show the 'Pricing' button"),
            # ELEMENT_GONE
            RunTest(url=f"{base_url}/modal", prompt="close the cookie banner"),
            # LIST_COUNT (infinite reveal)
            RunTest(url=f"{base_url}/infinite?total=150&chunk=25", prompt="load at least 100 results"),
            # LIST_ITEM_OPENED by index
            RunTest(url=f"{base_url}/list?n=30&prefix=Story", prompt="open the first item in the list", hints=RunSpec(
                goals=[
                    ListItemOpenedGoal(item=ItemSpec(position="first"), require_url_change=True)
                    ])),
            # LIST_ITEM_OPENED by content
            RunTest(url=f"{base_url}/list?n=40&prefix=Story", prompt="open the item that contains 'Story 17'"),
            # FILTER_APPLIED (chips appear as you check)
            RunTest(url=f"{base_url}/filters", prompt="filter by 'Remote' and 'iOS'"),
            # FIELD_VALUE_SET
            RunTest(url=f"{base_url}/form", prompt="set 'Location' to 'London'"),
            # FORM_COMPLETED + SUBMISSION_CONFIRMED (submit allowed by wording)
            RunTest(url=f"{base_url}/form", prompt="fill the required fields and submit the application"),
            # UPLOAD_ATTACHED (we supply assets via hints)
            RunTest(url=f"{base_url}/upload", prompt="upload my resume",
                 hints=RunSpec(
                     assets=Assets(
                         resume_path=resume_path,
                         upload_policy=UploadPolicy(allow_pdf=True, allow_docx=True, allow_images=True, max_mb=10)
                     )
                 )),
            # LOGIN_COMPLETED (we supply auth via hints)
            RunTest(url=f"{base_url}/login", prompt="log in with my credentials",
                 hints=RunSpec(auth=Auth(username="standard_user", password="secret_sauce"))),
            # NEW_TAB_OPENED
            RunTest(url=f"{base_url}/newtab", prompt="open 'Docs' in a new tab"),
            # REPEAT_UNTIL (wraps the previous list-count intent)
            RunTest(url=f"{base_url}/infinite?total=120&chunk=20", prompt="load results until we have 120 items"),
        ]

        # --- Selection filters -------------------------------------------------
        # --only supports comma-separated 1-based indices and ranges, e.g. "3" or "1,4-6"
        if only:
            idxs = set()
            for token in only.split(","):
                token = token.strip()
                if not token:
                    continue
                if "-" in token:
                    a, b = token.split("-", 1)
                    try:
                        a_i, b_i = int(a), int(b)
                    except ValueError:
                        continue
                    lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
                    idxs.update(range(lo, hi + 1))
                else:
                    try:
                        idxs.add(int(token))
                    except ValueError:
                        pass
            original = tests
            tests = [original[i - 1] for i in sorted(idxs) if 1 <= i <= len(original)]

        # --match filters by substring in prompt or url (case-insensitive)
        if match:
            m = match.lower()
            tests = [t for t in tests if m in t.prompt.lower() or m in t.url.lower()]

        if not tests:
            print("No tests selected. Adjust --only / --match filters.")
            ctx.close(); browser.close()
            return
        # ----------------------------------------------------------------------

        for i, t in enumerate(tests, 1):
            print("\n" + "="*80)
            print(f"TEST {i}: {t.prompt}  →  {t.url}")
            page.goto(t.url, wait_until="domcontentloaded")
            time.sleep(0.6)
            if t.hints:
                # vc.run(prompt=t.prompt, subprompt_hints=None, strict=False)  # let your infer work too
                # when hints are present, pass them as manual hints for the whole prompt:
                vc._exec_compressed_phrase(
                    prompt=t.prompt,
                    meta_instructions="Avoid destructive or final actions unless explicitly instructed.",
                    attempts=40,
                    hints=t.hints,
                )
            else:
                vc.run(prompt=t.prompt)

        print("\nAll tests dispatched. Manually verify the pages if needed.")
        ctx.close(); browser.close()

if __name__ == "__main__":
    # CLI flags: --only, --match, --headless
    parser = argparse.ArgumentParser(description="Run playground UI-goal tests.")
    parser.add_argument("--only", help="Comma-separated 1-based indices or ranges (e.g., 3 or 1,4-6).", default=None)
    parser.add_argument("--match", help="Substring to select tests by prompt or URL (case-insensitive).", default=None)
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    args = parser.parse_args()

    # Run the Flask server in a background process
    os.environ.setdefault("FLASK_ENV", "production")
    srv = multiprocessing.Process(target=_run_server, daemon=True)
    srv.start()
    time.sleep(1.0)  # small warmup

    try:
        run_tests(only=args.only, match=args.match, headless=args.headless)
    finally:
        srv.terminate()
