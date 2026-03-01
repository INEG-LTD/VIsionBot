from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Tuple

import markdown as md
from playwright.sync_api import sync_playwright
from pypdf import PdfReader


@dataclass(frozen=True)
class FitSettings:
    font_pt: float
    line_height: float
    margin_mm: int
    scale: float
    pages: int


def markdown_to_pdf_playwright(
    markdown_input: Union[str, os.PathLike],
    output_pdf: Union[str, os.PathLike],
    *,
    is_text: Optional[bool] = None,
    title: str = "Document",
    page_size: str = "A4",
    css_extra: str = "",
    # fitting
    max_pages: Optional[int] = None,
    guardrails: bool = True,
    write_best_effort: bool = False,
    margins_mm: Sequence[int] = (10, 8, 6, 4, 2, 0),
    font_pts: Sequence[float] = (11.5, 11.2, 11.0, 10.7, 10.4, 10.1, 9.9),
    line_heights: Sequence[float] = (1.40, 1.36, 1.34, 1.32, 1.30, 1.28, 1.26),
    scale_step: float = 0.03,
) -> Tuple[Path, FitSettings]:
    """
    Markdown -> PDF via Playwright.

    If max_pages is set, tries combinations of:
      - margin_mm in `margins_mm`
      - font_pt in `font_pts`
      - line_height in `line_heights`
      - then decreases Playwright `scale` until fit

    You control margins by passing: margins_mm=(1,2,3,...) etc.
    """
    out_path = Path(output_pdf).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Decide if input is text or a file path
    if is_text is None:
        treat_as_text = not Path(markdown_input).expanduser().exists()
    else:
        treat_as_text = bool(is_text)

    if treat_as_text:
        md_text = str(markdown_input)
        base_dir = Path.cwd()
    else:
        md_path = Path(markdown_input).expanduser().resolve()
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        md_text = md_path.read_text(encoding="utf-8")
        base_dir = md_path.parent

    html_body = md.markdown(
        md_text,
        extensions=["extra", "sane_lists", "tables", "fenced_code"],
        output_format="html5",
    )
    base_href = base_dir.resolve().as_uri().rstrip("/") + "/"

    def make_html(font_pt: float, line_height: float, margin_mm: int) -> str:
        css = f"""
@page {{
  size: {page_size};
  margin: {margin_mm}mm;
}}
html, body {{
  margin: 0 !important;
  padding: 0 !important;
  max-width: none !important;
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: {font_pt}pt;
  line-height: {line_height};
  color: #111;
}}
h1 {{ font-size: 1.55em; margin: 0 0 0.35em; line-height: 1.05; }}
h2 {{ font-size: 1.12em; margin: 0.85em 0 0.30em; border-bottom: 1px solid #ddd; padding-bottom: 0.16em; }}
h3 {{ font-size: 1.02em; margin: 0.65em 0 0.28em; }}
p  {{ margin: 0.28em 0; }}
ul, ol {{ margin: 0.28em 0 0.28em 1.10em; padding: 0; }}
li {{ margin: 0.16em 0; }}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  padding: 0.45em 0.55em;
  border-radius: 8px;
  background: #f6f6f6;
  margin: 0.30em 0;
}}
{css_extra}
"""
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <base href="{base_href}" />
  <style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    # Guardrails
    if guardrails:
        min_font_pt, min_line_height, min_margin_mm, min_scale = 9.8, 1.15, 0, 0.82
    else:
        min_font_pt, min_line_height, min_margin_mm, min_scale = 6.0, 1.0, 0, 0.10

    # Single render if no fitting requested
    if max_pages is None:
        chosen_margin = int(margins_mm[0]) if margins_mm else 10
        chosen_font = float(font_pts[0]) if font_pts else 11.0
        chosen_lh = float(line_heights[0]) if line_heights else 1.35

        html_doc = make_html(chosen_font, chosen_lh, chosen_margin)
        with tempfile.TemporaryDirectory() as td:
            html_file = Path(td) / "doc.html"
            html_file.write_text(html_doc, encoding="utf-8")
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(html_file.as_uri(), wait_until="networkidle")
                page.emulate_media(media="print")
                page.pdf(path=str(out_path), print_background=True, prefer_css_page_size=True)
                browser.close()

        return out_path, FitSettings(chosen_font, chosen_lh, chosen_margin, 1.0, 1)

    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")

    best: Optional[FitSettings] = None
    best_bytes: Optional[bytes] = None

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        html_file = td / "doc.html"
        pdf_tmp = td / "out.pdf"

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()

            for margin_mm in margins_mm:
                margin_mm = int(margin_mm)
                if margin_mm < min_margin_mm:
                    continue

                for font_pt in font_pts:
                    font_pt = float(font_pt)
                    if font_pt < min_font_pt:
                        continue

                    for line_h in line_heights:
                        line_h = float(line_h)
                        if line_h < min_line_height:
                            continue

                        html_doc = make_html(font_pt, line_h, margin_mm)
                        html_file.write_text(html_doc, encoding="utf-8")

                        page.goto(html_file.as_uri(), wait_until="networkidle")
                        page.emulate_media(media="print")

                        scale = 1.0
                        while scale >= min_scale:
                            if pdf_tmp.exists():
                                pdf_tmp.unlink()

                            page.pdf(
                                path=str(pdf_tmp),
                                print_background=True,
                                prefer_css_page_size=True,
                                scale=scale,
                            )

                            pages = len(PdfReader(str(pdf_tmp)).pages)
                            current = FitSettings(font_pt, line_h, margin_mm, scale, pages)

                            if best is None or pages < best.pages or (pages == best.pages and scale > best.scale):
                                best = current
                                best_bytes = pdf_tmp.read_bytes()

                            if pages <= max_pages:
                                out_path.write_bytes(pdf_tmp.read_bytes())
                                browser.close()
                                return out_path, current

                            scale = round(scale - scale_step, 2)

            browser.close()

    if write_best_effort and best is not None and best_bytes is not None:
        out_path.write_bytes(best_bytes)
        return out_path, best

    raise RuntimeError(
        f"Could not fit into <= {max_pages} page(s). "
        f"Best attempt was {best.pages if best else 'unknown'} page(s). "
        "Try guardrails=False, write_best_effort=True, or trim content."
    )
    
# if __name__ == "__main__":
#     pdf = markdown_to_pdf_playwright("md.md", "cv.pdf", guardrails=False, max_pages=1, margins_mm=(6,))
#     print("Wrote:", pdf)