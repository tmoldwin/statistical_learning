"""Build a Google Slides–compatible deck from paper/draft.md figures.

Output: paper/slides/paper_figures.pptx
Import in Google Slides: File → Import slides → Upload.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT = REPO_ROOT / "paper" / "draft.md"
OUT_DIR = REPO_ROOT / "paper" / "slides"
OUT_PPTX = OUT_DIR / "paper_figures.pptx"

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
CAPTION_RE = re.compile(r"^\*\*Figure (\d+)\.\*\*\s*(.+)$", re.MULTILINE)
TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def parse_figures(draft_text: str) -> list[dict]:
    """Pair each draft image with figure number and alt-text title."""
    lines = draft_text.splitlines()
    figures: list[dict] = []
    i = 0
    while i < len(lines):
        m_img = IMG_RE.match(lines[i].strip())
        if not m_img:
            i += 1
            continue
        alt, rel_path = m_img.group(1), m_img.group(2)
        fig_num: int | None = None
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            m_cap = CAPTION_RE.match(lines[j].strip())
            if m_cap:
                fig_num = int(m_cap.group(1))
        figures.append({
            "num": fig_num if fig_num is not None else len(figures) + 1,
            "title": alt.strip() or f"Figure {fig_num or len(figures) + 1}",
            "path": (REPO_ROOT / "paper" / rel_path).resolve(),
        })
        i = j + 1 if j > i else i + 1
    return figures


def _add_title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle


def _fit_image_box(img_w: int, img_h: int, max_w: float, max_h: float) -> tuple[float, float]:
    if img_w <= 0 or img_h <= 0:
        return max_w, max_h
    scale = min(max_w / img_w, max_h / img_h)
    return img_w * scale, img_h * scale


def _add_figure_slide(
    prs: Presentation,
    *,
    title: str,
    image_path: Path,
) -> None:
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    margin_x = Inches(0.2)
    top_y = Inches(0.2)
    title_h = Inches(0.7)
    bottom_margin = Inches(0.15)
    slide_w = prs.slide_width
    slide_h = prs.slide_height

    title_box = slide.shapes.add_textbox(margin_x, top_y, slide_w - 2 * margin_x, title_h)
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(24)
    p.font.bold = True

    img_top = top_y + title_h + Inches(0.05)
    img_max_w = float(slide_w - 2 * margin_x)
    img_max_h = float(slide_h - img_top - bottom_margin)

    from PIL import Image

    with Image.open(image_path) as im:
        px_w, px_h = im.size
    disp_w, disp_h = _fit_image_box(px_w, px_h, img_max_w, img_max_h)
    left = (float(slide_w) - disp_w) / 2
    top = float(img_top) + (img_max_h - disp_h) / 2
    slide.shapes.add_picture(str(image_path), left, top, width=disp_w, height=disp_h)


def build_slides(figures: list[dict], *, title: str, subtitle: str) -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    _add_title_slide(prs, title, subtitle)

    missing: list[str] = []
    for fig in figures:
        path = fig["path"]
        if not path.is_file():
            missing.append(str(path.relative_to(REPO_ROOT)))
            continue
        _add_figure_slide(
            prs,
            title=str(fig["title"]),
            image_path=path,
        )

    if missing:
        print("missing images (skipped):")
        for m in missing:
            print(f"  {m}")

    return prs


def main() -> None:
    draft_text = DRAFT.read_text(encoding="utf-8")
    title_m = TITLE_RE.search(draft_text)
    paper_title = title_m.group(1) if title_m else "Paper figures"
    subtitle = "Generated from paper/draft.md · import into Google Slides via Upload"

    figures = parse_figures(draft_text)
    if not figures:
        print(f"no figures found in {DRAFT}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs = build_slides(figures, title=paper_title, subtitle=subtitle)
    prs.save(str(OUT_PPTX))
    n_slides = len(prs.slides) - 1
    print(f"wrote {OUT_PPTX} ({n_slides} figure slides + title)")


if __name__ == "__main__":
    main()
