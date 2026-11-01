# -*- coding: utf-8 -*-
"""
سَنَد / SANAD — مساعد الامتثال لنظام حماية البيانات الشخصية السعودي (PDPL)
"امتثالٌ بدليل"

واجهة Streamlit احترافية تربط:
  • سؤال/جواب مؤسَّس على نصوص PDPL  →  rag_inference.query_copilot
  • تحليل فجوات سياسة خصوصية جهة     →  gap_analyser_inference.analyse_policy_text

المزوّدات المدعومة: Ollama و Gemini فقط.

التشغيل:
    streamlit run app.py
"""
import os, base64, json, pathlib, traceback
import streamlit as st
from pdf_report import build_pdf

HERE = pathlib.Path(__file__).parent.resolve()


def _load_secrets_into_env():
    """يقرأ الإعدادات من .streamlit/secrets.toml (أو Secrets في Hugging Face) إلى متغيرات البيئة."""
    try:
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "OLLAMA_MODEL", "OLLAMA_HOST", "STAGE_DATA_DIR"):
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass  # لا يوجد ملف secrets: مقبول


_load_secrets_into_env()

# ──────────────────────────────────────────────────────────────────────────
# هوية سَنَد البصرية
# ──────────────────────────────────────────────────────────────────────────
GREEN_D = "#1f4d3f"   # أخضر عميق
GREEN_M = "#2e6e57"   # أخضر متوسط
NAVY_D  = "#16293b"   # كحلي غامق
NAVY_M  = "#1b2b40"
GOLD    = "#c6a15b"   # ذهبي
MINT    = "#cfe8dd"   # نعناعي فاتح
PAPER   = "#f6f4ee"

st.set_page_config(page_title="سَنَد — SANAD · مساعد امتثال PDPL",
                   page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")


def _logo_b64() -> str:
    for name in ("sanad_logo.jpg", "sanad_logo.png", "_logo_b64.txt"):
        p = HERE / "assets" / name
        if not p.exists():
            p = HERE / name
        if p.exists():
            if name.endswith(".txt"):
                return p.read_text().strip()
            return base64.b64encode(p.read_bytes()).decode()
    return ""


LOGO = _logo_b64()

# ──────────────────────────────────────────────────────────────────────────
# CSS — تصميم عربي RTL بهوية سَنَد
# ──────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
    direction: rtl;
}}
.stApp {{
    background:
      radial-gradient(1200px 600px at 100% -10%, {GREEN_M}22, transparent 60%),
      radial-gradient(900px 500px at -10% 110%, {GOLD}18, transparent 55%),
      {PAPER};
}}
#MainMenu, footer, header {{visibility: hidden;}}

/* ── الهيدر ── */
.sanad-hero {{
    display:flex; align-items:center; gap:22px;
    background: linear-gradient(135deg, {NAVY_D} 0%, {GREEN_D} 100%);
    border-radius: 22px; padding: 26px 30px; margin: 4px 0 10px 0;
    box-shadow: 0 18px 40px -18px {NAVY_D}cc;
    border: 1px solid {GOLD}44;
}}
.sanad-hero img {{
    width: 92px; height: 92px; border-radius: 18px; object-fit: cover;
    background: #ffffff10; padding: 6px; border: 1px solid {GOLD}55;
    box-shadow: 0 8px 22px -10px #000a;
}}
.sanad-hero .t-wrap {{ display:flex; flex-direction:column; gap:4px; }}
.sanad-hero h1 {{
    color:#fff; margin:0; font-size: 2.15rem; font-weight:700; letter-spacing:.5px;
}}
.sanad-hero .en {{ color:{GOLD}; font-weight:600; letter-spacing:3px; font-size:.95rem; }}
.sanad-hero .tag {{
    color:{MINT}; font-size:1.05rem; font-weight:500; margin-top:2px;
}}
.sanad-hero .tag b {{ color:{GOLD}; }}

/* ── شريط التنبيه القانوني ── */
.legal {{
    background: {GOLD}1a; border:1px solid {GOLD}66; border-radius:12px;
    padding:10px 16px; color:{NAVY_D}; font-size:.9rem; margin: 2px 0 14px 0;
}}
.legal b {{ color:{GREEN_D}; }}

/* ── البطاقات ── */
.card {{
    background:#fff; border:1px solid #0000000f; border-radius:18px;
    padding:20px 22px; box-shadow: 0 12px 30px -20px {NAVY_D}80; margin-bottom:14px;
}}

/* ── شارات الحالة ── */
.badge {{ display:inline-block; padding:3px 12px; border-radius:999px;
         font-weight:600; font-size:.82rem; margin:2px; }}
.b-cov  {{ background:#15803d22; color:#15803d; border:1px solid #15803d55; }}
.b-par  {{ background:#b4530022; color:#b45309; border:1px solid #b4530955; }}
.b-mis  {{ background:#b9191922; color:#b91919; border:1px solid #b9191955; }}
.b-rev  {{ background:{NAVY_D}18; color:{NAVY_D}; border:1px solid {NAVY_D}44; }}

/* ── رقائق الاستشهاد ── */
.cite {{ display:inline-block; background:{GREEN_D}14; color:{GREEN_D};
        border:1px solid {GREEN_D}44; border-radius:8px; padding:3px 10px;
        margin:3px; font-size:.84rem; font-weight:600; }}

/* ── مؤشرات KPI ── */
.kpi {{ background:linear-gradient(160deg,#fff, {MINT}55); border:1px solid {GREEN_D}22;
        border-radius:16px; padding:16px; text-align:center; }}
.kpi .n {{ font-size:2rem; font-weight:700; line-height:1; }}
.kpi .l {{ color:{NAVY_D}aa; font-size:.85rem; margin-top:6px; }}

.stButton>button {{
    background: linear-gradient(135deg,{GREEN_D},{GREEN_M}); color:#fff;
    border:none; border-radius:12px; padding:10px 22px; font-weight:600;
    box-shadow:0 10px 24px -12px {GREEN_D}; }}
.stButton>button:hover {{ filter:brightness(1.08); }}

.stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
.stTabs [data-baseweb="tab"] {{
    background:#fff; border:1px solid #00000010; border-radius:12px 12px 0 0;
    padding:8px 18px; font-weight:600; }}
.stTabs [aria-selected="true"] {{ background:{GREEN_D}; color:#fff; }}

.answer-box {{ background:{MINT}44; border-right:4px solid {GREEN_D};
              border-radius:0 12px 12px 0; padding:16px 18px; font-size:1.05rem;
              line-height:1.9; color:{NAVY_D}; direction: rtl; text-align: right; }}
.answer-box p {{ margin: 0 0 8px 0; }}
/* ── الشريط الجانبي: فاتح بنص داكن مقروء ── */
section[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,{MINT}99,{PAPER});
    border-left:1px solid {GREEN_D}22;
    min-width: 280px !important; }}
section[data-testid="stSidebar"] * {{ color:{NAVY_D} !important; }}
section[data-testid="stSidebar"] .stButton>button,
section[data-testid="stSidebar"] .stButton>button * {{ color:#fff !important; }}
section[data-testid="stSidebar"] .stButton>button {{
    background:linear-gradient(135deg,{GREEN_D},{GREEN_M}) !important; }}

/* ── ضمان القراءة ومنع أبيض على أبيض ── */
html, body, .stApp, .stMarkdown, p, span, label, li {{ color:{NAVY_D}; }}
.stTextInput input, .stTextArea textarea {{
    color:{NAVY_D} !important; -webkit-text-fill-color:{NAVY_D} !important;
    background:#fff !important; }}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {{
    color:{NAVY_D}88 !important; -webkit-text-fill-color:{NAVY_D}88 !important; }}
.stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] * {{ color:{NAVY_D} !important; }}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * {{ color:#fff !important; }}
.card {{ color:{NAVY_D}; }}
.answer-box, .answer-box * {{ color:{NAVY_D} !important; }}
.legal, .legal * {{ color:{NAVY_D} !important; }}
.legal b {{ color:{GREEN_D} !important; }}
.sanad-hero h1 {{ color:#fff !important; }}
.sanad-hero .en {{ color:{GOLD} !important; }}
.sanad-hero .tag {{ color:{MINT} !important; }}
.sanad-hero .tag b {{ color:{GOLD} !important; }}

/* توحيد اتجاه ومحاذاة النص العربي */
[data-testid="stAppViewContainer"] .main .block-container {{
    direction: rtl;
}}

[data-testid="stMarkdownContainer"] {{
    direction: rtl !important;
    text-align: right !important;
}}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    direction: rtl !important;
    text-align: right !important;
}}

/* حقول الأسئلة والإدخال */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {{
    direction: rtl !important;
    text-align: right !important;
}}

/* عناوين الحقول */
[data-testid="stWidgetLabel"] {{
    direction: rtl !important;
    text-align: right !important;
}}

/* محاذاة الإجابة */
.answer-box {{
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: plaintext;
}}

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# الهيدر
# ──────────────────────────────────────────────────────────────────────────
logo_img = f'<img src="data:image/jpeg;base64,{LOGO}"/>' if LOGO else ""
st.markdown(f"""
<div class="sanad-hero">
  {logo_img}
  <div class="t-wrap">
    <div class="en">S&nbsp;A&nbsp;N&nbsp;A&nbsp;D</div>
    <h1>سَنَد</h1>
    <div class="tag">مساعد الامتثال لنظام حماية البيانات الشخصية — <b>امتثالٌ بدليل</b></div>
  </div>
</div>
<div class="legal">
  ⚖️ <b>تنبيه:</b> سَنَد أداة مساعدة تستند إلى نصوص النظام ولوائحه، وليست استشارة قانونية.
  كل إجابة مؤسَّسة على مادة رسمية مُستشهَد بها؛ يرجى الرجوع لمختصٍّ قانوني للقرارات الملزمة.
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# الشريط الجانبي — الإعدادات (Ollama / Gemini فقط)
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    provider = st.radio("مزوّد النموذج اللغوي",
                        ["Ollama (محلي)", "Gemini (سحابي)"], index=0)
    is_gemini = provider.startswith("Gemini")


    if is_gemini:
        gkey = os.getenv("GEMINI_API_KEY", "")
        gmodel = st.text_input(
            "نموذج Gemini",
            value="models/gemini-3.5-flash"
        )
    else:
        omodel = st.text_input("نموذج Ollama", value="qwen2.5:7b")
        ohost = st.text_input("Ollama Host", value="http://localhost:11434")


    st.divider()
    data_dir = st.text_input("مجلد بيانات RAG (STAGE_DATA_DIR)",
                             value=os.getenv("STAGE_DATA_DIR", str(HERE / "data")),
                             help="يحتوي مخرجات المرحلتين 1 و2 (الفهارس والمقاطع).")
    use_rag = st.toggle("تأسيس التحليل على نصوص PDPL (RAG)", value=True)

    st.divider()
    if st.button("🔌 الاتصال / تهيئة النظام", use_container_width=True):
        st.session_state.pop("_ready", None)
        st.session_state["_connect"] = True

    st.caption("سَنَد · مشروع تخرّج AIEP — المسار B")


def _apply_env():
    """يضبط متغيّرات البيئة من إعدادات الواجهة قبل تحميل الأنبوب."""
    os.environ["STAGE_DATA_DIR"] = data_dir
    if is_gemini:
        os.environ["LLM_PROVIDER"] = "gemini"
        if gkey:
            os.environ["GEMINI_API_KEY"] = gkey
            os.environ["GOOGLE_API_KEY"] = gkey
        os.environ["GEMINI_MODEL"] = gmodel
        return "gemini"
    else:
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_MODEL"] = omodel
        os.environ["OLLAMA_HOST"] = ohost
        return "ollama"


@st.cache_resource(show_spinner=False)
def _load_rag(provider_key: str, data_dir_key: str):
    import rag_inference as rag
    info = rag.load_pipeline(provider_key)
    return rag, info


def _ensure_ready():
    if st.session_state.get("_ready"):
        return True
    if not st.session_state.get("_connect"):
        return False
    prov = _apply_env()
    with st.spinner("جارٍ تهيئة النظام (الفهارس + النموذج)…"):
        try:
            rag, info = _load_rag(prov, data_dir)
            st.session_state["_rag"] = rag
            st.session_state["_ready"] = True
            st.session_state["_prov"] = prov
            st.success("تم تهيئة سَنَد بنجاح ✅")
            return True
        except Exception as e:
            st.error(f"تعذّرت التهيئة: {type(e).__name__}: {e}")
            with st.expander("تفاصيل"):
                st.code(traceback.format_exc())
            return False


# ──────────────────────────────────────────────────────────────────────────
# التبويبات
# ──────────────────────────────────────────────────────────────────────────
tab_qa, tab_gap, tab_about = st.tabs(
    ["💬 سؤال وجواب", "🔍 تحليل امتثال سياسة", "ℹ️ عن سَنَد"])

STATUS_AR = {"covered": "مُغطّى", "partial": "جزئي", "missing": "مفقود",
             "needs_review": "يحتاج مراجعة"}
STATUS_CLS = {"covered": "b-cov", "partial": "b-par", "missing": "b-mis",
              "needs_review": "b-rev"}


def status_badge(s):
    return f'<span class="badge {STATUS_CLS.get(s,"b-rev")}">{STATUS_AR.get(s,s)}</span>'


import re as _re, html as _html

DOC_NAMES = {"pdpl": "نظام حماية البيانات الشخصية",
             "implementing_regulation": "اللائحة التنفيذية"}
_CITE_RE = _re.compile(r"\[\s*(?:المصدر|Source)\s*:\s*([^|\]]+?)\s*\|\s*(?:المادة|Article)\s*:\s*([^\]]+?)\s*\]")


def render_answer(text: str) -> str:
    """يحوّل وسوم الاستشهاد إلى شارات، ويهرب النص أولًا، ويعرض الإجابة من اليمين لليسار."""
    def badge(m):
        doc = DOC_NAMES.get(m.group(1).strip(), m.group(1).strip())
        art = m.group(2).strip()
        label = doc if art.upper() == "N/A" else f"{doc} · المادة {art}"
        return f'<span class="cite">📎 {label}</span>'
    paras = []
    for raw in (text or "").splitlines():
        line = _html.escape(raw.strip())
        if line:
            paras.append(f"<p>{_CITE_RE.sub(badge, line)}</p>")
    return '<div class="answer-box">' + "".join(paras) + "</div>"


# ───── تبويب السؤال/الجواب ─────
with tab_qa:
    st.markdown("#### اسأل عن نظام حماية البيانات الشخصية ولائحته التنفيذية")
    q = st.text_area("سؤالك", height=90,
                     placeholder="مثال: ما هي حقوق صاحب البيانات الشخصية؟ · ما تعريف «البيانات الحسّاسة»؟")
    ask = st.button("اسأل سَنَد", key="ask")

    if ask:
        if not q.strip():
            st.warning("اكتب سؤالًا أولًا.")
        elif not _ensure_ready():
            st.info("اضغط «الاتصال / تهيئة النظام» من الشريط الجانبي أولًا.")
        else:
            rag = st.session_state["_rag"]
            with st.spinner("يبحث في النصوص الرسمية ويؤسّس الإجابة…"):
                try:
                    res = rag.query_copilot(q.strip())
                except Exception as e:
                    st.error(f"خطأ أثناء الاستعلام: {type(e).__name__}: {e}")
                    res = None
            if res:
                abst = res.get("abstained")
                ans = res.get("answer", "") or ""
                st.markdown('<div class="card">', unsafe_allow_html=True)
                if abst:
                    st.markdown(f'{status_badge("missing")} '
                                '<b>امتناع:</b> لا يوجد أساس رسمي كافٍ للإجابة.',
                                unsafe_allow_html=True)
                st.markdown(render_answer(ans), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                with st.expander("المخرجات الخام (JSON)"):
                    st.json(res)


# ───── تبويب تحليل الامتثال ─────
with tab_gap:
    st.markdown("#### حلّل سياسة خصوصية جهة مقابل متطلبات PDPL")
    c1, c2 = st.columns([2, 1])
    with c1:
        entity = st.text_input("اسم الجهة", placeholder="مثال: شركة س للتجارة")
    with c2:
        up = st.file_uploader("أو ارفع ملف السياسة", type=["txt", "md", "json"])
    policy_text = st.text_area("نص السياسة", height=200,
                               placeholder="الصق نص سياسة الخصوصية هنا…")
    analyse = st.button("حلّل الامتثال", key="analyse")

    if analyse:
        raw = policy_text.strip()
        if up is not None and not raw:
            raw = up.read().decode("utf-8", errors="ignore")
        if not raw:
            st.warning("الصق نص السياسة أو ارفع ملفًا.")
        elif not _ensure_ready():
            st.info("اضغط «الاتصال / تهيئة النظام» من الشريط الجانبي أولًا.")
        else:
            prov = st.session_state.get("_prov", "ollama")
            with st.spinner("يصنّف البنود، يسترجع الأساس الرسمي، ويحكم على التغطية…"):
                try:
                    import gap_analyser_inference as gi
                    report = gi.analyse_policy_text(
                        raw, entity_name=entity or (up.name if up else "سياسة"),
                        provider=prov, use_rag=use_rag)
                except Exception as e:
                    st.error(f"خطأ أثناء التحليل: {type(e).__name__}: {e}")
                    with st.expander("تفاصيل"):
                        st.code(traceback.format_exc())
                    report = None

            if report:
                s = report["summary"]
                cols = st.columns(5)
                kpis = [("التغطية %", f'{report.get("coverage_pct",0)}', GREEN_D),
                        ("مُغطّى", s.get("covered", 0), "#15803d"),
                        ("جزئي", s.get("partial", 0), "#b45309"),
                        ("مفقود", s.get("missing", 0), "#b91919"),
                        ("البنود", report.get("n_clauses", 0), NAVY_D)]
                for col, (lbl, val, clr) in zip(cols, kpis):
                    col.markdown(
                        f'<div class="kpi"><div class="n" style="color:{clr}">{val}</div>'
                        f'<div class="l">{lbl}</div></div>', unsafe_allow_html=True)

                mode = report.get("judge_mode", "")
                if "heuristic" in mode or "no LLM" in mode:
                    st.markdown('<div class="legal">ℹ️ وضع تقديري (بلا LLM): النتائج '
                                'مبدئية. فعّل التأسيس (RAG) واتصل بمزوّد لحكمٍ نهائي.</div>',
                                unsafe_allow_html=True)

                st.markdown("##### تقرير الفجوات")
                for it in report["gap_report"]:
                    stt = it["status"]
                    with st.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(
                            f'{status_badge(stt)} <b>{it["requirement_id"]}</b> — '
                            f'{it["requirement"]}', unsafe_allow_html=True)
                        basis = it.get("pdpl_basis", {})
                        arts = basis.get("articles", {})
                        art_txt = []
                        if arts.get("pdpl"):
                            art_txt.append("نظام: " + ", ".join(map(str, arts["pdpl"])))
                        if arts.get("impl"):
                            art_txt.append("لائحة: " + ", ".join(map(str, arts["impl"])))
                        if art_txt:
                            st.markdown(" ".join(f'<span class="cite">📎 {a}</span>'
                                                 for a in art_txt), unsafe_allow_html=True)
                        if it.get("entity_evidence"):
                            st.markdown(f'<div class="answer-box" style="font-size:.95rem">'
                                        f'«{it["entity_evidence"]}»</div>',
                                        unsafe_allow_html=True)
                        if it.get("rationale"):
                            st.caption("📝 " + it["rationale"])
                        st.markdown('</div>', unsafe_allow_html=True)

                st.download_button(
                    "⬇️ تنزيل التقرير (JSON)",
                    data=json.dumps(report, ensure_ascii=False, indent=2),
                    file_name=f'sanad_gap_{(entity or "report").replace(" ","_")}.json',
                    mime="application/json")
                st.download_button(
                    "⬇️ تنزيل التقرير (PDF)",
                    data=build_pdf(report, entity=entity),
                    file_name=f'sanad_gap_{(entity or "report").replace(" ","_")}.pdf',
                    mime="application/pdf")


# ───── تبويب التعريف ─────
with tab_about:
    st.markdown(f"""
<div class="card">
<h3 style="color:{GREEN_D};margin-top:0">عن سَنَد</h3>
<p style="line-height:1.9;color:{NAVY_D}">
<b>سَنَد</b> مساعد امتثال لنظام حماية البيانات الشخصية السعودي (PDPL) ولائحته التنفيذية.
يقوم على ثلاث طبقات: استرجاع هجين مؤسَّس على النصوص الرسمية، وتحقّق استشهاد يمنع الهلوسة،
وامتناع واعٍ عند غياب الأساس. ويضيف محلّل فجوات يقيس سياسة أي جهة مقابل متطلبات النظام
(مُغطّى / جزئي / مفقود) بأدلّة من الطرفين: نص PDPL، وبند سياسة الجهة.
</p>
<ul style="line-height:2;color:{NAVY_D}">
  <li>🔎 <b>سؤال/جواب مؤسَّس</b> — إجابات باستشهاد للمواد، أو امتناع صريح.</li>
  <li>🧭 <b>تحليل فجوات</b> — 23 متطلبًا من النظام واللائحة مع خريطة تصنيف البنود.</li>
  <li>🔐 <b>مزوّدات:</b> Ollama (محلي) و Gemini (سحابي).</li>
</ul>
<p style="color:{NAVY_D}99;font-size:.85rem">
مشروع تخرّج AIEP — المسار B · "امتثالٌ بدليل"
</p>
</div>
""", unsafe_allow_html=True)
