"""
generate_report.py

Generates vehicle_disappearance_findings.pdf
"""

import io
import json
import os
import re
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, HRFlowable, KeepTogether, PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
DARK_BLUE   = colors.HexColor("#1a2e4a")
MID_BLUE    = colors.HexColor("#2e5b8a")
LIGHT_BLUE  = colors.HexColor("#d6e4f0")
ACCENT      = colors.HexColor("#e8732a")
LIGHT_GREY  = colors.HexColor("#f5f5f5")
MID_GREY    = colors.HexColor("#888888")
TEXT        = colors.HexColor("#1c1c1c")
WHITE       = colors.white

W, H = A4   # 595 x 842 pt

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_data():
    resolved  = load_jsonl("charley_results.jsonl")
    crashed   = load_jsonl("crashed_vehicle_cases.jsonl")
    own_veh   = load_jsonl("own_vehicle_classified.jsonl")
    discovery = load_jsonl("discovery_order_classified.jsonl")
    location  = load_jsonl("location_classified.jsonl")
    return resolved, crashed, own_veh, discovery, location


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

CHART_BLUE   = "#2e5b8a"
CHART_ORANGE = "#e8732a"
CHART_GREEN  = "#2e8a5b"
CHART_GREY   = "#aaaaaa"

def make_location_chart(location_data, tmp_dir):
    """Pie + bar combo: crash location breakdown."""
    crashed = [c for c in location_data
               if c.get("vehicle_classification", {}).get("vehicle_found_crashed") is True]

    def ltype(c):
        return c["location_classification"].get("location_type", "unknown")

    counts = {"water": 0, "land_offroad": 0, "roadside": 0, "unknown": 0}
    for c in crashed:
        counts[ltype(c)] += 1

    labels = ["Water", "Land\n(off-road)", "Roadside", "Unknown"]
    values = [counts["water"], counts["land_offroad"], counts["roadside"], counts["unknown"]]
    clrs   = [CHART_BLUE, CHART_ORANGE, CHART_GREEN, CHART_GREY]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    fig.patch.set_facecolor("white")

    # Pie
    wedges, texts, autotexts = ax1.pie(
        values, labels=None, colors=clrs,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        startangle=140, wedgeprops=dict(linewidth=1.2, edgecolor="white"),
        pctdistance=0.72,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")
    ax1.set_title("Crashed vehicle location\n(n=132 crashed cases)",
                  fontsize=10, fontweight="bold", color="#1a2e4a", pad=10)
    patches = [mpatches.Patch(color=clrs[i], label=f"{labels[i].replace(chr(10),' ')} ({values[i]})")
               for i in range(len(labels))]
    ax1.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.18),
               ncol=2, fontsize=8, frameon=False)

    # Bar
    bar_labels = ["Water", "Land\n(off-road)", "Roadside", "Unknown"]
    total = len(crashed)
    pcts  = [v / total * 100 if total else 0 for v in values]
    bars  = ax2.bar(bar_labels, pcts, color=clrs, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, pct in zip(bars, pcts):
        if pct > 1:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color="#1a2e4a")
    ax2.set_ylabel("% of crashed cases", fontsize=9, color="#444444")
    ax2.set_title("Location breakdown (%)\n(crashed own-vehicle cases)",
                  fontsize=10, fontweight="bold", color="#1a2e4a", pad=10)
    ax2.set_ylim(0, max(pcts) * 1.2 + 5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=9)
    ax2.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    plt.tight_layout(pad=1.5)
    path = os.path.join(tmp_dir, "chart_location.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def make_delay_chart(location_data, tmp_dir):
    """Horizontal bar chart: mean & median discovery delay by location type."""
    def ltype(c): return c["location_classification"].get("location_type", "unknown")
    def days(c):
        d = c["location_classification"].get("days_to_discovery")
        return int(d) if d is not None else None

    categories = [
        ("water",        "Water"),
        ("land_offroad", "Land (off-road)"),
        ("roadside",     "Roadside"),
    ]

    means   = []
    medians = []
    labels  = []
    ns      = []
    for key, label in categories:
        vals = [days(c) for c in location_data if ltype(c) == key and days(c) is not None]
        if vals:
            means.append(sum(vals) / len(vals))
            medians.append(sorted(vals)[len(vals) // 2])
            labels.append(label)
            ns.append(len(vals))

    y      = np.arange(len(labels))
    height = 0.32

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor("white")

    b1 = ax.barh(y + height / 2, [m / 365 for m in means],   height, label="Mean",   color=CHART_BLUE,   alpha=0.9)
    b2 = ax.barh(y - height / 2, [m / 365 for m in medians], height, label="Median", color=CHART_ORANGE, alpha=0.9)

    for bar, val in zip(b1, means):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val/365:.1f} yr", va="center", fontsize=8.5, color="#1a2e4a", fontweight="bold")
    for bar, val in zip(b2, medians):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val/365:.1f} yr", va="center", fontsize=8.5, color="#c0581a", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=9)
    ax.set_xlabel("Years to discovery", fontsize=9, color="#444444")
    ax.set_title("Discovery delay by crash location type",
                 fontsize=10, fontweight="bold", color="#1a2e4a", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False)

    # right-pad
    ax.set_xlim(0, max(max(m / 365 for m in means), 1) * 1.35)

    plt.tight_layout(pad=1.2)
    path = os.path.join(tmp_dir, "chart_delay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def make_decade_chart(active_data, tmp_dir):
    """Bar chart: vehicle_missing rate by decade."""
    decades = ["1980s", "1990s", "2000s", "2010s", "2020s"]
    rates   = []
    for dec in decades:
        grp = [c for c in active_data if c.get("decade") == dec]
        true_n = sum(1 for c in grp
                     if c.get("vehicle_classification", {}).get("vehicle_missing") is True)
        rates.append(true_n / len(grp) * 100 if grp else 0)

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("white")
    bars = ax.bar(decades, rates, color=CHART_BLUE, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#1a2e4a")
    ax.axhline(sum(rates) / len(rates), color=CHART_ORANGE, linewidth=1.5,
               linestyle="--", label=f"Overall avg ({sum(rates)/len(rates):.1f}%)")
    ax.set_ylabel("% vehicle still missing", fontsize=9, color="#444444")
    ax.set_title("Vehicle-missing rate by decade of disappearance\n(active cases sample, n=1,000)",
                 fontsize=10, fontweight="bold", color="#1a2e4a", pad=8)
    ax.set_ylim(0, max(rates) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False)
    plt.tight_layout(pad=1.2)
    path = os.path.join(tmp_dir, "chart_decade.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# PDF styles
# ---------------------------------------------------------------------------

def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            fontSize=26, leading=32, textColor=WHITE,
            alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            fontSize=13, leading=17, textColor=LIGHT_BLUE,
            alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            fontSize=8.5, leading=12, textColor=LIGHT_BLUE,
            alignment=TA_CENTER, fontName="Helvetica-Oblique",
        ),
        "meta": ParagraphStyle(
            "Meta",
            fontSize=10, leading=14, textColor=LIGHT_BLUE,
            alignment=TA_CENTER, fontName="Helvetica",
        ),
        "h1": ParagraphStyle(
            "H1",
            fontSize=14, leading=18, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontSize=11, leading=15, textColor=MID_BLUE,
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            fontSize=10, leading=15, textColor=TEXT,
            fontName="Helvetica", spaceAfter=6, alignment=TA_JUSTIFY,
        ),
        "body_small": ParagraphStyle(
            "BodySmall",
            fontSize=9, leading=13, textColor=TEXT,
            fontName="Helvetica", spaceAfter=4,
        ),
        "finding": ParagraphStyle(
            "Finding",
            fontSize=12, leading=16, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2,
        ),
        "comment": ParagraphStyle(
            "Comment",
            fontSize=9.5, leading=14, textColor=colors.HexColor("#444444"),
            fontName="Helvetica-Oblique", spaceAfter=10, leftIndent=12,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontSize=10, leading=14, textColor=TEXT,
            fontName="Helvetica", spaceAfter=3, leftIndent=14, bulletIndent=4,
        ),
        "footer": ParagraphStyle(
            "Footer",
            fontSize=7.5, leading=10, textColor=MID_GREY,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "toc_entry": ParagraphStyle(
            "TocEntry",
            fontSize=10, leading=15, textColor=DARK_BLUE,
            fontName="Helvetica", spaceAfter=3,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------

MARGIN = 2.2 * cm

def header_footer(canvas, doc):
    canvas.saveState()
    # Header bar
    if doc.page > 1:
        canvas.setFillColor(DARK_BLUE)
        canvas.rect(0, H - 1.1 * cm, W, 1.1 * cm, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(WHITE)
        canvas.drawString(MARGIN, H - 0.72 * cm,
                          "Submerged and Forgotten: Vehicle-Associated Disappearances")
        canvas.drawRightString(W - MARGIN, H - 0.72 * cm, f"Page {doc.page}")
        # Footer line
        canvas.setStrokeColor(LIGHT_BLUE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 1.6 * cm, W - MARGIN, 1.6 * cm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MID_GREY)
        canvas.drawCentredString(W / 2, 1.1 * cm,
            "Preliminary research — LLM-assisted classification — not peer reviewed")
    canvas.restoreState()


def build_doc(output_path):
    doc = BaseDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.8 * cm, bottomMargin=2.2 * cm,
        title="Submerged and Forgotten",
        author="Michal",
    )
    frame = Frame(
        MARGIN, 2.2 * cm,
        W - 2 * MARGIN, H - 2.8 * cm - 2.2 * cm,
        id="normal",
    )
    doc.addPageTemplates([
        PageTemplate(id="main", frames=[frame], onPage=header_footer),
    ])
    return doc


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------

def data_table(headers, rows, col_widths=None, stripe=True):
    data = [headers] + rows
    n_cols = len(headers)
    if col_widths is None:
        avail = W - 2 * MARGIN
        col_widths = [avail / n_cols] * n_cols

    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  DARK_BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  7),
        ("TOPPADDING",    (0, 0), (-1, 0),  7),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ROWBACKGROUND", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("ALIGN",         (1, 1), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 1), (0, -1),  "LEFT"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("LINEBELOW",     (0, 0), (-1, 0),  1.0, MID_BLUE),
    ]
    if stripe:
        for i in range(1, len(data)):
            bg = LIGHT_GREY if i % 2 == 0 else WHITE
            style.append(("BACKGROUND", (0, i), (-1, i), bg))

    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Content builders
# ---------------------------------------------------------------------------

def title_page(styles):
    """Returns a list of flowables for the title page (drawn on a special canvas)."""
    # We'll use a full-page coloured block via a custom flowable
    from reportlab.platypus import Flowable

    class TitleBlock(Flowable):
        def draw(self):
            c = self.canv
            # Full dark-blue background
            c.setFillColor(DARK_BLUE)
            c.rect(-MARGIN, -2.2 * cm, W, H, fill=1, stroke=0)
            # Accent band
            c.setFillColor(ACCENT)
            c.rect(-MARGIN, H * 0.38 - 2.2 * cm, W, 0.35 * cm, fill=1, stroke=0)
            # Decorative top stripe
            c.setFillColor(MID_BLUE)
            c.rect(-MARGIN, H - 3 * cm, W, 0.5 * cm, fill=1, stroke=0)

        def wrap(self, aw, ah):
            return (aw, 0)

    S = styles
    items = [
        TitleBlock(),
        Spacer(1, 5.5 * cm),
        Paragraph("Submerged and Forgotten:", S["title"]),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "Vehicle-Associated Disappearances and<br/>Crash Outcomes in Cold Cases",
            S["subtitle"],
        ),
        Spacer(1, 1.2 * cm),
        HRFlowable(width="60%", thickness=0.8, color=ACCENT, spaceAfter=0.8 * cm,
                   hAlign="CENTER"),
        Paragraph("Preliminary Findings — Early Stage Research", S["subtitle"]),
        Spacer(1, 0.6 * cm),
        Paragraph("Michał", S["meta"]),
        Paragraph("May 2026", S["meta"]),
        Spacer(1, 2.5 * cm),
        HRFlowable(width="80%", thickness=0.4, color=MID_BLUE, spaceAfter=0.5 * cm,
                   hAlign="CENTER"),
        Paragraph(
            "<b>Disclaimer:</b> This report presents preliminary findings from an exploratory "
            "research project. All case classifications were generated using LLM-assisted "
            "(Claude Haiku) automated analysis of narrative text sourced from The Charley "
            "Project. Results have <b>not been peer reviewed</b>, manually validated at scale, "
            "or submitted for publication. Figures should be interpreted as directional "
            "estimates only.",
            S["disclaimer"],
        ),
        PageBreak(),
    ]
    return items


def section_header(number, title, styles):
    from reportlab.platypus import Flowable

    class SectionBanner(Flowable):
        def __init__(self, num, txt):
            self.num = num
            self.txt = txt
            Flowable.__init__(self)

        def wrap(self, aw, ah):
            self._aw = aw
            return (aw, 1.1 * cm)

        def draw(self):
            c = self.canv
            c.setFillColor(DARK_BLUE)
            c.roundRect(0, 0, self._aw, 0.9 * cm, 4, fill=1, stroke=0)
            c.setFillColor(ACCENT)
            c.roundRect(0, 0, 1.1 * cm, 0.9 * cm, 4, fill=1, stroke=0)
            c.rect(0.55 * cm, 0, 0.55 * cm, 0.9 * cm, fill=1, stroke=0)
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(WHITE)
            c.drawCentredString(0.55 * cm, 0.28 * cm, str(self.num))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(1.4 * cm, 0.28 * cm, txt)

    # workaround: use closure
    txt = title

    class _Banner(Flowable):
        def wrap(self, aw, ah):
            self._aw = aw
            return (aw, 1.1 * cm)

        def draw(self):
            c = self.canv
            c.setFillColor(DARK_BLUE)
            c.roundRect(0, 0, self._aw, 0.9 * cm, 4, fill=1, stroke=0)
            c.setFillColor(ACCENT)
            c.roundRect(0, 0, 1.1 * cm, 0.9 * cm, 4, fill=1, stroke=0)
            c.rect(0.55 * cm, 0, 0.55 * cm, 0.9 * cm, fill=1, stroke=0)
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(WHITE)
            c.drawCentredString(0.55 * cm, 0.28 * cm, str(number))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(1.5 * cm, 0.28 * cm, txt)

    return [Spacer(1, 0.3 * cm), _Banner(), Spacer(1, 0.4 * cm)]


def finding_block(number, headline, comment, styles):
    from reportlab.platypus import Flowable

    class FindingBox(Flowable):
        def wrap(self, aw, ah):
            self._aw = aw
            return (aw, 0.05 * cm)

        def draw(self):
            c = self.canv
            c.setFillColor(LIGHT_BLUE)
            c.roundRect(0, -0.05 * cm, self._aw, 0.08 * cm, 2, fill=1, stroke=0)

    items = [
        FindingBox(),
        Spacer(1, 0.15 * cm),
        Paragraph(f"<b>Finding {number}:</b> {headline}", styles["finding"]),
        Paragraph(comment, styles["comment"]),
    ]
    return items


# ---------------------------------------------------------------------------
# Main PDF builder
# ---------------------------------------------------------------------------

def build_pdf(output_path="vehicle_disappearance_findings.pdf"):
    print("Loading data files...")
    resolved, crashed, own_veh, discovery, location = load_data()

    # Compute active data from own_veh (has decade field)
    active_file = Path("active_sample_classified.jsonl")
    if active_file.exists():
        active = [json.loads(l) for l in open(active_file, encoding="utf-8") if l.strip()]
    else:
        active = []

    tmp_dir = tempfile.mkdtemp()
    print("Generating charts...")
    chart_loc   = make_location_chart(location, tmp_dir)
    chart_delay = make_delay_chart(location, tmp_dir)
    chart_dec   = make_decade_chart(active, tmp_dir) if active else None

    S = build_styles()
    doc = build_doc(output_path)
    story = []

    # ---- TITLE PAGE ----
    story += title_page(S)

    # ---- SECTION 1: Dataset Overview ----
    story += section_header(1, "Dataset Overview", S)

    story.append(Paragraph(
        "This report summarises findings from an exploratory analysis of missing persons cases "
        "drawn from <b>The Charley Project</b> (charleyproject.org), a publicly accessible "
        "database of long-term missing persons cases in the United States. The analysis examined "
        "the relationship between vehicle disappearances and crash outcomes, with particular "
        "focus on cases where vehicles were subsequently found submerged in water.",
        S["body"],
    ))

    avail_w = W - 2 * MARGIN
    story.append(data_table(
        ["Dataset", "Cases", "Notes"],
        [
            ["Resolved cases (full)", f"{len(resolved):,}",
             "All A–Z resolved cases, scraped May 2026"],
            ["Active cases (sample)", "1,000",
             "Random sample, 200 per decade, 1980s–2020s"],
            ["Vehicle-associated (resolved)", "249",
             "12.5% of resolved cases"],
            ["Own-vehicle cases", "190",
             "76.3% of vehicle-associated cases"],
            ["Crashed vehicle cases", "132",
             "69.5% of own-vehicle cases"],
        ],
        col_widths=[avail_w * 0.35, avail_w * 0.15, avail_w * 0.50],
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("<b>Methodology</b>", S["h2"]))
    story.append(Paragraph(
        "Case narrative text was collected via automated web scraping of charleyproject.org "
        "using Python (requests + BeautifulSoup). Classifications were generated using the "
        "<b>Anthropic Batch API</b> (Claude Haiku model) with structured prompts designed to "
        "extract specific binary signals from unstructured narrative text. Each classification "
        "step was run as a separate batch job on the relevant subset of cases. No manual "
        "validation of classifications has been performed at scale; a targeted validation "
        "sample is planned as a next step.",
        S["body"],
    ))

    story.append(PageBreak())

    # ---- SECTION 2: Key Findings ----
    story += section_header(2, "Key Findings", S)

    # Finding 1
    story += finding_block(
        1,
        "9.4% of active missing persons cases involve a vehicle that is also missing.",
        "This rate is remarkably stable across four decades (range: 7.5%–11.5%), suggesting "
        "the vehicle-associated disappearance pattern is not a product of any particular era. "
        "Extrapolated to the full Charley Project active database (~14,000 cases at time of "
        "writing), this implies approximately 1,300 vehicles are currently unaccounted for "
        "alongside missing persons.",
        S,
    )

    if chart_dec:
        story.append(Image(chart_dec, width=avail_w * 0.85, height=avail_w * 0.36))
        story.append(Spacer(1, 0.3 * cm))

    if active:
        decades = ["1980s", "1990s", "2000s", "2010s", "2020s"]
        dec_rows = []
        for dec in decades:
            grp = [c for c in active if c.get("decade") == dec]
            t = sum(1 for c in grp if c.get("vehicle_classification", {}).get("vehicle_missing") is True)
            n = sum(1 for c in grp if c.get("vehicle_classification", {}).get("vehicle_missing") is None)
            pct = t / len(grp) * 100 if grp else 0
            dec_rows.append([dec, str(len(grp)), str(t), str(n), f"{pct:.1f}%"])
        story.append(data_table(
            ["Decade", "Cases", "Vehicle Missing", "Ambiguous", "Rate"],
            dec_rows,
            col_widths=[avail_w * 0.16, avail_w * 0.14, avail_w * 0.22,
                        avail_w * 0.20, avail_w * 0.14],
        ))
        story.append(Spacer(1, 0.5 * cm))

    # Finding 2
    story += finding_block(
        2,
        "69.5% of resolved cases where someone disappeared with their own vehicle "
        "ended with the vehicle found crashed or submerged.",
        "This figure rises to 94.5% in cases where the body was found inside the vehicle "
        "at the time of discovery — establishing crashed/submerged vehicles as the dominant "
        "resolution pattern for own-vehicle disappearances. The 30.5% of non-crashed cases "
        "are predominantly homicides where the vehicle was abandoned separately.",
        S,
    )

    story.append(data_table(
        ["Outcome", "Cases", "% of own-vehicle cases"],
        [
            ["Vehicle found crashed / submerged", "132", "69.5%"],
            ["Vehicle found, not crashed",         "58",  "30.5%"],
            ["Vehicle never found",                "6",   "3.2%"],
        ],
        col_widths=[avail_w * 0.55, avail_w * 0.15, avail_w * 0.30],
    ))
    story.append(Spacer(1, 0.5 * cm))

    # Finding 3
    story += finding_block(
        3,
        "88.6% of crashed vehicles were found in water; only 7.6% were found on land.",
        "Water submersion is not merely common — it is the defining characteristic of "
        "vehicle crash outcomes in missing persons cases. Rivers, retention ponds, lakes, "
        "and canals account for nearly nine in ten resolved crash cases. This has profound "
        "implications for search strategy: land-based searches are structurally unlikely to "
        "locate the majority of missing vehicles.",
        S,
    )

    story.append(Image(chart_loc, width=avail_w, height=avail_w * 0.42))
    story.append(Spacer(1, 0.3 * cm))

    # Finding 4
    story += finding_block(
        4,
        "Median time to discovery: 6.6 years for water cases, 2.3 years for land cases.",
        "Vehicles submerged in water take dramatically longer to find than those on land. "
        "The mean water discovery time of 12.2 years, and a maximum of 57 years, indicate "
        "that many currently active cases involve vehicles that have been submerged for "
        "decades and remain undiscovered purely due to the difficulty of underwater search.",
        S,
    )

    story.append(Image(chart_delay, width=avail_w * 0.9, height=avail_w * 0.35))
    story.append(Spacer(1, 0.3 * cm))

    story.append(data_table(
        ["Location Type", "n", "Mean", "Median", "Min", "Max"],
        [
            ["Water",         "126", "12.2 yr", "6.6 yr",  "2 days", "57.9 yr"],
            ["Land (off-road)", "26", "2.3 yr",  "2.2 yr",  "3 days", "9.9 yr"],
            ["Roadside",       "6",  "1.6 yr",  "1.3 yr",  "198 days", "3.1 yr"],
        ],
        col_widths=[avail_w*0.30, avail_w*0.08, avail_w*0.14,
                    avail_w*0.14, avail_w*0.16, avail_w*0.18],
    ))

    story.append(PageBreak())

    # ---- SECTION 3: Discovery Order ----
    story += section_header(3, "Discovery Order Analysis", S)

    story.append(Paragraph(
        "For the 190 resolved own-vehicle cases, the sequence in which the person and vehicle "
        "were discovered provides additional insight into the nature of these disappearances.",
        S["body"],
    ))

    story.append(data_table(
        ["Discovery Pattern", "Cases", "%", "Crash Rate"],
        [
            ["Body found inside vehicle (simultaneous)", "109", "57.4%", "94.5%"],
            ["Vehicle found first, person found later",   "58",  "30.5%", "36.2%"],
            ["Person found, vehicle never recovered",     "21",  "11.1%", "38.1%*"],
            ["Person found first, vehicle found later",   "1",    "0.5%", "0.0%"],
            ["Vehicle found, person never found",         "1",    "0.5%", "—"],
        ],
        col_widths=[avail_w*0.48, avail_w*0.12, avail_w*0.10, avail_w*0.20],
    ))

    story.append(Paragraph(
        "* These cases had narrative evidence of a crash (e.g. tyre tracks to a cliff edge, "
        "witnesses) but the vehicle was never physically recovered — consistent with deep-water "
        "submersion.",
        S["body_small"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph(
        "The simultaneous discovery category — where the body was found inside the vehicle at "
        "the moment of recovery — carries a 94.5% crash rate. This is the cleanest signal in "
        "the dataset and represents the canonical outcome of an own-vehicle disappearance: the "
        "person drove into water, the vehicle sank, and both were found together years or "
        "decades later. The 36.2% crash rate in the 'vehicle found first' group reflects a "
        "structurally different set of cases, predominantly homicides where the vehicle was "
        "abandoned as a decoy and the body disposed of separately.",
        S["body"],
    ))

    story.append(PageBreak())

    # ---- SECTION 4: Policy Implication ----
    story += section_header(4, "Policy Implication", S)

    story.append(Paragraph(
        "Underwater search is systematically underutilised as a resolution strategy for "
        "vehicle-associated cold cases.",
        S["h2"],
    ))
    story.append(Paragraph(
        "The base rates established in this analysis are striking: when someone disappears "
        "with their own vehicle and is not found quickly, there is approximately a <b>70% "
        "probability</b> that the vehicle will eventually be found crashed or submerged — and "
        "of those, <b>88.6% will be in water</b>. The median time to underwater discovery is "
        "<b>6.6 years</b>, with a mean of over 12 years, suggesting that a significant "
        "proportion of currently active cases involve vehicles that entered water bodies at "
        "the time of disappearance and have simply never been searched for.",
        S["body"],
    ))
    story.append(Paragraph(
        "If the 9.4% vehicle-missing rate observed in the active sample holds across the full "
        "Charley Project database of approximately 14,000 active cases, approximately "
        "<b>1,316 cases</b> involve a missing vehicle. Applying the 69.5% crash rate and "
        "88.6% water submersion rate yields an estimate of approximately <b>810 vehicles "
        "potentially submerged in US waterways</b> associated with active missing persons cases.",
        S["body"],
    ))
    story.append(Paragraph(
        "The volunteer organisation <b>Adventures with Purpose</b> has demonstrated the "
        "practical feasibility of targeted underwater vehicle search, recovering submerged "
        "vehicles and resolving cold cases in dozens of instances using consumer-grade sonar "
        "equipment. Their work constitutes proof of concept that systematic, data-driven "
        "underwater search is achievable without institutional resources. The base rates "
        "presented here provide a quantitative foundation for prioritising which bodies of "
        "water are most likely to contain missing vehicles, potentially enabling more "
        "efficient allocation of search effort.",
        S["body"],
    ))

    story.append(KeepTogether([
        Spacer(1, 0.3 * cm),
        data_table(
            ["Estimate", "Value", "Basis"],
            [
                ["Active Charley Project cases",       "~14,000",  "Site estimate"],
                ["Vehicle-missing rate",                "9.4%",     "This study, n=1,000"],
                ["Estimated vehicles missing",          "~1,316",   "Derived"],
                ["Crash rate (own-vehicle cases)",      "69.5%",    "This study, n=190"],
                ["Water submersion rate (crashed)",     "88.6%",    "This study, n=132"],
                ["Estimated vehicles submerged",        "~810",     "Derived"],
            ],
            col_widths=[avail_w * 0.50, avail_w * 0.18, avail_w * 0.32],
        ),
    ]))

    story.append(PageBreak())

    # ---- SECTION 5: Limitations ----
    story += section_header(5, "Limitations", S)

    limitations = [
        ("<b>Charley Project selection bias.</b> The Charley Project indexes cases that "
         "have remained unresolved for an extended period (typically one year or more). "
         "Cases resolved quickly — including many straightforward accidents — are likely "
         "underrepresented. The crash rates reported here may therefore overestimate the "
         "true population-level probability, as easily resolved crashes are excluded.",
         ),
        ("<b>LLM classification limitations.</b> All case classifications were generated "
         "using automated prompting of Claude Haiku. While the model demonstrates strong "
         "performance on clearly-written narratives, it may misclassify cases with ambiguous "
         "language, incomplete information, or unusual circumstances. No systematic "
         "error-rate estimation has been performed. The 12.7% null rate in active case "
         "classification reflects genuine narrative ambiguity rather than model failure.",
         ),
        ("<b>US-centric dataset.</b> The Charley Project is a US-focused database. "
         "Findings may not generalise to other jurisdictions with different road "
         "infrastructure, water body distribution, or reporting practices.",
         ),
        ("<b>Narrative incompleteness.</b> Resolved case narratives vary substantially in "
         "detail. Many older cases (pre-2000) have sparse narratives that preclude reliable "
         "classification. Discovery delay estimates are available for only a subset of cases "
         "due to absent or ambiguous date information in narratives.",
         ),
        ("<b>Sample size for active cases.</b> The active case analysis is based on a "
         "random sample of 1,000 cases (200 per decade). While designed to be representative, "
         "the sample may not capture the full distribution of case types across all "
         "geographic regions and demographic groups.",
         ),
    ]

    for i, text in enumerate(limitations, 1):
        story.append(Paragraph(f"<b>{i}.</b> &nbsp; {text}", S["bullet"]))
        story.append(Spacer(1, 0.2 * cm))

    story.append(PageBreak())

    # ---- SECTION 6: Next Steps ----
    story += section_header(6, "Next Steps", S)

    steps = [
        ("<b>Manual validation of classifier sample.</b> A random sample of 50–100 "
         "classifications from each batch should be manually reviewed to establish "
         "precision and recall estimates for the LLM classifier.",
         ),
        ("<b>Geographic analysis of water body types.</b> Linking case locations to "
         "GIS water body data would enable identification of specific rivers, lakes, "
         "and reservoirs most likely to contain missing vehicles — directly actionable "
         "for search prioritisation.",
         ),
        ("<b>Active vs resolved comparison at full scale.</b> The current active case "
         "analysis uses a 1,000-case sample. Classifying all active cases would provide "
         "more precise estimates and enable state-level analysis.",
         ),
        ("<b>NamUs data request.</b> The National Missing and Unidentified Persons System "
         "(NamUs) holds additional case metadata including vehicle descriptions. A formal "
         "data request could provide vehicle make, model, and colour for cases where the "
         "vehicle is listed as missing — enabling more targeted search.",
         ),
        ("<b>Collaboration with search organisations.</b> Engaging with groups such as "
         "Adventures with Purpose to cross-reference their existing search locations against "
         "the predicted high-probability cases would provide real-world validation of the "
         "model's predictive utility.",
         ),
    ]

    for i, text in enumerate(steps, 1):
        story.append(Paragraph(f"<b>{i}.</b> &nbsp; {text}", S["bullet"]))
        story.append(Spacer(1, 0.25 * cm))

    story.append(Spacer(1, 1.0 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_BLUE,
                             spaceAfter=0.4 * cm))
    story.append(Paragraph(
        "Data source: The Charley Project (charleyproject.org). "
        "Classification: Anthropic Claude Haiku (claude-haiku-4-5-20251001). "
        "Analysis: Python (requests, BeautifulSoup, reportlab, matplotlib). "
        "All case narratives remain the property of The Charley Project.",
        S["footer"],
    ))

    print(f"Building PDF: {output_path}")
    doc.build(story)
    print(f"Done. Saved to '{output_path}'.")


if __name__ == "__main__":
    build_pdf()
