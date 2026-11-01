# -*- coding: utf-8 -*-
"""تصدير تقرير تحليل الامتثال إلى PDF بالعربي (RTL) — سَنَد."""
import io, os, datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import arabic_reshaper
from bidi.algorithm import get_display

HERE = Path(__file__).parent
FONT_CANDIDATES = [
    HERE / "assets" / "fonts" / "Amiri-Regular.ttf",
    HERE / "assets" / "fonts" / "DejaVuSans.ttf",
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]
STATUS_AR = {"covered": "مُغطّى", "partial": "جزئي", "missing": "مفقود", "needs_review": "يحتاج مراجعة"}
STATUS_BG = {"covered": "#D9EAD3", "partial": "#FFF2CC", "missing": "#F4CCCC", "needs_review": "#E7E6E6"}
NAVY = colors.HexColor("#16293b")
GREEN = colors.HexColor("#1f4d3f")


def _font():
    for p in FONT_CANDIDATES:
        if p.exists():
            pdfmetrics.registerFont(TTFont("SanadAR", str(p)))
            return "SanadAR"
    raise FileNotFoundError("لم يُعثر على خط يدعم العربية")


def _has_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in text)


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ar(text) -> str:
    """نص عربي ليس فيه إلا سطر واحد (عناوين، شارات، أرقام)."""
    text = "" if text is None else str(text)
    if not _has_arabic(text):
        return _xml(text)
    return _xml(get_display(arabic_reshaper.reshape(text), base_dir="R"))


def ar_wrap(text, width, font, size) -> str:
    """يقطع النص العربي إلى أسطر بنفسه، ثم يرتّب كل سطر على حدة.
    بهذي الطريقة ما يتقلب ترتيب الكلمات بين الأسطر."""
    text = "" if text is None else str(text)
    if not _has_arabic(text):
        return _xml(text)
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if pdfmetrics.stringWidth(arabic_reshaper.reshape(trial), font, size) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "<br/>".join(_xml(get_display(arabic_reshaper.reshape(l), base_dir="R")) for l in lines)


def build_pdf(report: dict, entity: str = "") -> bytes:
    font = _font()
    styles = {
        "title": ParagraphStyle("t", fontName=font, fontSize=18, leading=24, alignment=1, textColor=GREEN),
        "sub": ParagraphStyle("s", fontName=font, fontSize=10, leading=14, alignment=1, textColor=NAVY),
        "h": ParagraphStyle("h", fontName=font, fontSize=12, leading=18, alignment=2, textColor=NAVY),
        "cell": ParagraphStyle("c", fontName=font, fontSize=8.5, leading=12, alignment=2),
        "small": ParagraphStyle("sm", fontName=font, fontSize=8, leading=11, alignment=2, textColor=colors.grey),
    }
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm,
                            title="SANAD Gap Analysis Report")
    el = []
    name = entity or report.get("entity", "")
    el.append(Paragraph(ar("سَنَد — تقرير تحليل الامتثال لنظام حماية البيانات الشخصية"), styles["title"]))
    el.append(Paragraph(ar(f"الجهة: {name}"), styles["sub"]))
    el.append(Paragraph(ar(f"التاريخ: {datetime.date.today().isoformat()}"), styles["sub"]))
    el.append(Spacer(1, 0.5*cm))

    s = report.get("summary", {})
    kpi = [["التغطية %", "مُغطّى", "جزئي", "مفقود", "البنود"],
           [str(report.get("coverage_pct", 0)), str(s.get("covered", 0)), str(s.get("partial", 0)),
            str(s.get("missing", 0)), str(report.get("n_clauses", ""))]]
    kpi = [[ar(c) for c in row] for row in kpi]
    kt = Table(kpi, colWidths=[3.3*cm]*5)
    kt.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), font, 10),
        ("BACKGROUND", (0, 0), (-1, 0), GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EEF5F2")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    el.append(kt)
    el.append(Spacer(1, 0.6*cm))

    el.append(Paragraph(ar("تفاصيل المتطلبات"), styles["h"]))
    head = [ar(h) for h in ["الأساس النظامي", "الدليل من النص", "الحالة", "المتطلب"]]
    data = [head]
    bgs = []
    for i, it in enumerate(report.get("gap_report", []), start=1):
        arts = it.get("pdpl_basis", {}).get("articles", {})
        art_txt = []
        if arts.get("pdpl"):
            art_txt.append("نظام: " + ", ".join(map(str, arts["pdpl"])))
        if arts.get("impl"):
            art_txt.append("لائحة: " + ", ".join(map(str, arts["impl"])))
        ev = it.get("entity_evidence") or "—"
        rat = it.get("rationale") or ""
        req = f'{it.get("requirement_id","")} — {it.get("requirement", it.get("requirement_en",""))}'
        status = STATUS_AR.get(it.get("status"), it.get("status", ""))
        cw = 7.0 * cm  # عرض خانة الدليل تقريبًا (بعد الهوامش)
        ev_en = not _has_arabic(ev)
        ev_cell = (_xml(ev[:400] + ("…" if len(ev) > 400 else "")) if ev_en
                   else ar_wrap(ev[:400], cw, font, 8.5))
        rat_cell = ar_wrap(rat, cw, font, 8.5) if rat else ""
        data.append([
            Paragraph(ar_wrap(" | ".join(art_txt) or "—", 3.3*cm, font, 8.5), styles["cell"]),
            Paragraph(ev_cell + (("<br/>" + rat_cell) if rat_cell else ""), styles["cell"]),
            Paragraph(ar(status), styles["cell"]),
            Paragraph(_xml(req) if not _has_arabic(req) else ar_wrap(req, 4.3*cm, font, 8.5), styles["cell"]),
        ])
        bgs.append(("BACKGROUND", (2, i), (2, i), colors.HexColor(STATUS_BG.get(it.get("status"), "#FFFFFF"))))
    t = Table(data, colWidths=[3.6*cm, 7.6*cm, 2.2*cm, 4.6*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), font, 9), ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ] + bgs))
    el.append(t)
    el.append(Spacer(1, 0.5*cm))
    el.append(Paragraph(ar("تنبيه: هذا التقرير استرشادي مستخلص من النصوص الرسمية، وليس استشارة قانونية."),
                        styles["small"]))
    doc.build(el)
    return buf.getvalue()
