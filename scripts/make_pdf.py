"""Render the Markdown design document to PDF.

    python -m scripts.make_pdf docs/DESIGN.md docs/DESIGN.pdf

The brief asks for the design report as a PDF. Rather than keep two copies of
the same content in different formats — which drift the moment one is edited —
the Markdown is the single source and this generates the PDF from it.

Deliberately a narrow converter, not a general one. It handles exactly the
constructs DESIGN.md uses: ATX headings, paragraphs, bullet and numbered lists,
fenced code blocks, block quotes, pipe tables, horizontal rules, and inline
bold/italic/code/links. Anything else passes through as text. A full Markdown
implementation would be far more code for output nobody would read differently.

reportlab is used because it is pure Python: no wkhtmltopdf, no headless
browser, no system packages, so `pip install -r requirements-dev.txt` is enough
to reproduce the deliverable on any machine.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#1f6feb")
INK = colors.HexColor("#111418")
MUTED = colors.HexColor("#5b6570")
RULE = colors.HexColor("#d8dee6")
CODE_BG = colors.HexColor("#f4f6f8")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
        leading=14, textColor=INK, spaceAfter=7, alignment=TA_LEFT,
    )
    return {
        "body": body,
        "h1": ParagraphStyle("H1", parent=body, fontName="Helvetica-Bold", fontSize=19,
                             leading=24, spaceBefore=4, spaceAfter=10, textColor=INK),
        "h2": ParagraphStyle("H2", parent=body, fontName="Helvetica-Bold", fontSize=13.5,
                             leading=18, spaceBefore=16, spaceAfter=7, textColor=ACCENT),
        "h3": ParagraphStyle("H3", parent=body, fontName="Helvetica-Bold", fontSize=11,
                             leading=15, spaceBefore=11, spaceAfter=5, textColor=INK),
        "h4": ParagraphStyle("H4", parent=body, fontName="Helvetica-BoldOblique", fontSize=9.5,
                             leading=13, spaceBefore=9, spaceAfter=4, textColor=MUTED),
        "code": ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=7.8,
                               leading=10.2, textColor=INK, backColor=CODE_BG,
                               borderPadding=6, spaceBefore=4, spaceAfter=9),
        "quote": ParagraphStyle("Quote", parent=body, fontName="Helvetica-Oblique",
                                fontSize=9, leading=13, textColor=MUTED,
                                leftIndent=10, borderPadding=0, spaceAfter=8),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=8.2, leading=11, spaceAfter=0),
        "cellh": ParagraphStyle("CellH", parent=body, fontSize=8.2, leading=11,
                                fontName="Helvetica-Bold", spaceAfter=0),
    }


_INLINE = [
    (re.compile(r"`([^`]+)`"), r'<font face="Courier" size="8.4">\1</font>'),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), r"<i>\1</i>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<link href="\2" color="#1f6feb">\1</link>'),
]


def inline(text: str) -> str:
    """Markdown inline spans to reportlab's mini-HTML.

    Escaping happens first: the source contains `<` and `&` in prose and code,
    and reportlab would otherwise treat them as markup and fail to parse the
    paragraph.
    """
    out = html.escape(text, quote=False)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    # Escaping turned the arrows in tables into entities; restore readable ones.
    return out.replace("--&gt;", "→").replace("-&gt;", "→")


def parse_table(lines: list[str], st: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for raw in lines:
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        rows.append(cells)
    # Row 1 of a pipe table is the alignment separator.
    if len(rows) > 1 and all(set(c) <= set("-: ") for c in rows[1]):
        aligns = rows[1]
        rows.pop(1)
    else:
        aligns = ["---"] * len(rows[0])

    width = A4[0] - 36 * mm
    ncols = max(len(r) for r in rows)
    data = []
    for i, row in enumerate(rows):
        row = row + [""] * (ncols - len(row))
        style = st["cellh"] if i == 0 else st["cell"]
        data.append([Paragraph(inline(c), style) for c in row])

    table = Table(data, colWidths=[width / ncols] * ncols, repeatRows=1, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f7f9fb")),
    ]
    for col, spec in enumerate(aligns[:ncols]):
        if spec.endswith(":") and not spec.startswith(":"):
            commands.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
        elif spec.startswith(":") and spec.endswith(":"):
            commands.append(("ALIGN", (col, 0), (col, -1), "CENTER"))
    table.setStyle(TableStyle(commands))
    return table


def convert(markdown: str) -> list:
    st = styles()
    flow: list = []
    lines = markdown.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # fenced code
        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            # Keep short blocks whole rather than splitting them across pages.
            code = Preformatted("\n".join(block), st["code"])
            flow.append(KeepTogether(code) if len(block) <= 24 else code)
            continue

        # table
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            flow.append(Spacer(1, 3))
            flow.append(parse_table(block, st))
            flow.append(Spacer(1, 9))
            continue

        # headings
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            key = {1: "h1", 2: "h2", 3: "h3"}.get(level, "h4")
            if level == 2:
                # Avoid a heading stranded at the foot of a page.
                flow.append(CondPageBreak(48))
            flow.append(Paragraph(inline(text), st[key]))
            if level == 1:
                flow.append(HRFlowable(width="100%", thickness=1, color=ACCENT,
                                       spaceBefore=2, spaceAfter=10))
            i += 1
            continue

        if stripped.startswith("---") and set(stripped) <= {"-"}:
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                   spaceBefore=8, spaceAfter=10))
            i += 1
            continue

        if stripped.startswith(">"):
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(block)), st["quote"]))
            continue

        # lists
        bullet = re.match(r"^[-*]\s+(.*)", stripped)
        number = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if bullet or number:
            ordered = number is not None
            items = []
            while i < len(lines):
                current = lines[i].strip()
                m2 = re.match(r"^(\d+)\.\s+(.*)" if ordered else r"^[-*]\s+(.*)", current)
                if not m2:
                    # A continuation line is indented under the previous item.
                    if current and lines[i].startswith(("  ", "\t")) and items:
                        items[-1] = items[-1] + " " + current
                        i += 1
                        continue
                    break
                items.append(m2.group(2) if ordered else m2.group(1))
                i += 1
            flow.append(
                ListFlowable(
                    [ListItem(Paragraph(inline(t), st["body"]), leftIndent=14)
                     for t in items],
                    bulletType="1" if ordered else "bullet",
                    bulletFontSize=8,
                    leftIndent=14,
                    bulletColor=MUTED,
                )
            )
            flow.append(Spacer(1, 5))
            continue

        # paragraph: join until a blank line or a construct starts
        block = []
        while i < len(lines):
            current = lines[i].strip()
            if not current or current.startswith(("#", "|", "```", ">", "---")):
                break
            if re.match(r"^[-*]\s+", current) or re.match(r"^\d+\.\s+", current):
                break
            block.append(current)
            i += 1
        flow.append(Paragraph(inline(" ".join(block)), st["body"]))

    return flow


def build(source: Path, target: Path, title: str, subtitle: str) -> None:
    doc = BaseDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=title, author="Comparables.ai technical assessment",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")

    def decorate(canvas, document):  # noqa: ANN001 - reportlab callback signature
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, 10 * mm, subtitle)
        canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"{document.page}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.3)
        canvas.line(doc.leftMargin, 13 * mm, A4[0] - doc.rightMargin, 13 * mm)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(convert(source.read_text(encoding="utf-8")))


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/DESIGN.md")
    target = Path(sys.argv[2] if len(sys.argv) > 2 else source.with_suffix(".pdf"))
    if not source.exists():
        print(f"not found: {source}", file=sys.stderr)
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    build(
        source,
        target,
        title=f"Company Search — {source.stem.title()}",
        subtitle="Company Search · Comparables.ai technical assessment",
    )
    print(f"wrote {target} ({target.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
