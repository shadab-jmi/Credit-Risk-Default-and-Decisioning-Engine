"""Credit-Risk Decision Engine — a small Streamlit scorer.

Look up a held-out applicant by ID, or score a new applicant from a short form.
Either way you get a calibrated PD, an approve/decline call at the cost-optimal
threshold t*, and the top-3 reasons. Run from the project root:

    streamlit run app/streamlit_app.py
"""
import os
import sys

import streamlit as st

# make the project root importable and current whether launched from root or app/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
os.chdir(ROOT)

from src import scoring

st.set_page_config(page_title="Credit-Risk Decision Engine", page_icon="💳",
                   layout="centered")

MODE_LOOKUP = "Look up existing applicant"
MODE_NEW = "Score a new applicant"

# maps a chosen history to the model's late-payment rate feature
MISSED_PAYMENT_OPTIONS = {
    "None — always paid on time": 0.0,
    "Some — occasionally late": 0.15,
    "Many — frequently late": 0.40,
}


# inline-SVG icons (lucide-style); stroke=currentColor inherits container colour
def _svg(paths, w=2.0):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
            f"fill='none' stroke='currentColor' stroke-width='{w}' "
            f"stroke-linecap='round' stroke-linejoin='round'>{paths}</svg>")

ICON_LOGO = _svg("<path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z'/>"
                 "<path d='m9 12 2 2 4-4'/>")
ICON_CHECK = _svg("<path d='M20 6 9 17l-5-5'/>", w=2.6)
ICON_X = _svg("<path d='M18 6 6 18'/><path d='m6 6 12 12'/>", w=2.6)
ICON_ALERT = _svg("<path d='m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 "
                  "2 0 0 0 1.7-3Z'/><path d='M12 9v4'/><path d='M12 17h.01'/>")
ICON_INFO = _svg("<circle cx='12' cy='12' r='10'/><path d='M12 16v-4'/>"
                 "<path d='M12 8h.01'/>")
ICON_MONEY = _svg("<rect width='20' height='12' x='2' y='6' rx='2'/>"
                  "<circle cx='12' cy='12' r='2'/><path d='M6 12h.01M18 12h.01'/>")
ICON_USER = _svg("<path d='M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2'/>"
                 "<circle cx='12' cy='7' r='4'/>")
ICON_CARD = _svg("<rect width='20' height='14' x='2' y='5' rx='2'/>"
                 "<line x1='2' x2='22' y1='10' y2='10'/>")

# URL-encoded SVG masks for the two segmented-toggle options
_SEARCH_MASK = ("%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
                "fill='none' stroke='%23000' stroke-width='2.4' stroke-linecap='round' "
                "stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E"
                "%3Cpath d='m21 21-4.3-4.3'/%3E%3C/svg%3E")
_USERPLUS_MASK = ("%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
                  "fill='none' stroke='%23000' stroke-width='2.2' stroke-linecap='round' "
                  "stroke-linejoin='round'%3E%3Cpath d='M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 "
                  "0 0-4 4v2'/%3E%3Ccircle cx='9' cy='7' r='4'/%3E%3Cpath d='M19 8v6'/%3E"
                  "%3Cpath d='M22 11h-6'/%3E%3C/svg%3E")


# all custom styling lives in this one block (config.toml sets the base theme)
CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
  --bg-base:#0B0D12; --bg-card:#141821; --bg-elev:#1C2230;
  --border:rgba(255,255,255,.07); --border-strong:rgba(255,255,255,.14);
  --text:#E6E9F0; --muted:#8B93A7;
  --accent:#6366F1; --accent-2:#818CF8; --accent-glow:rgba(99,102,241,.42);
  --ok:#22C55E; --ok-glow:rgba(34,197,94,.30);
  --bad:#F43F5E; --bad-glow:rgba(244,63,94,.30);
  --r-sm:8px; --r:12px; --r-lg:18px;
  --shadow:0 10px 34px rgba(0,0,0,.45);
}}

/* ---- base surfaces & type ---- */
.stApp {{
  background:
    radial-gradient(900px 520px at 10% -12%, rgba(99,102,241,.13), transparent 60%),
    radial-gradient(760px 520px at 100% -6%, rgba(34,197,94,.05), transparent 55%),
    var(--bg-base);
  color:var(--text);
  font-family:'Inter',system-ui,-apple-system,sans-serif;
}}
/* padding-top clears Streamlit's fixed top toolbar */
.block-container {{max-width:900px; padding-top:5rem; padding-bottom:3.5rem;}}
h1,h2,h3 {{font-family:'Space Grotesk','Inter',sans-serif; letter-spacing:-.01em; color:var(--text);}}
hr {{border-color:var(--border);}}

/* ---- app header ---- */
.app-header {{display:flex; align-items:center; gap:18px; margin:.2rem 0 1.7rem;}}
.app-logo {{width:46px; height:46px; border-radius:13px; display:flex; align-items:center;
  justify-content:center; background:linear-gradient(145deg,var(--accent-2),var(--accent));
  box-shadow:0 10px 26px var(--accent-glow); flex:0 0 46px;}}
.app-logo svg {{width:24px; height:24px; color:#0B0D12;}}
.app-heading {{min-width:0; flex:1 1 auto;}}
.app-title {{font-family:'Space Grotesk'; font-weight:700; font-size:1.5rem;
  letter-spacing:-.02em; line-height:1.15; white-space:normal; overflow-wrap:break-word;}}
.app-sub {{color:var(--muted); font-size:.9rem; margin-top:3px;}}
.app-pill {{margin-left:auto; font-size:.7rem; font-weight:600; letter-spacing:.07em;
  color:var(--accent-2); background:rgba(99,102,241,.12); border:1px solid rgba(99,102,241,.3);
  padding:6px 12px; border-radius:999px; white-space:nowrap;}}
.app-pill .dot {{color:var(--ok);}}

/* ---- segmented mode toggle (restyled radio) ---- */
.seg-label {{color:var(--muted); font-size:.75rem; font-weight:600; letter-spacing:.08em;
  text-transform:uppercase; margin:.4rem 0 .5rem;}}
[data-testid="stRadio"] > div[role="radiogroup"] {{display:inline-flex; gap:6px;
  background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:6px;}}
[data-testid="stRadio"] [role="radiogroup"] label {{display:flex; align-items:center; gap:9px;
  margin:0 !important; padding:10px 20px; border-radius:9px; color:var(--muted);
  font-weight:500; cursor:pointer; transition:all .15s ease;}}
[data-testid="stRadio"] [role="radiogroup"] label:hover {{color:var(--text); background:var(--bg-elev);
  box-shadow:0 3px 12px rgba(0,0,0,.28);}}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{
  background:var(--accent); color:#0B0D12; font-weight:600; box-shadow:0 6px 18px var(--accent-glow);}}
[data-testid="stRadio"] [role="radiogroup"] label > div:first-of-type {{display:none !important;}}
[data-testid="stRadio"] [role="radiogroup"] label::before {{content:""; width:16px; height:16px;
  flex:0 0 16px; background:currentColor; -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
  -webkit-mask-position:center; mask-position:center; -webkit-mask-size:contain; mask-size:contain;}}
[data-testid="stRadio"] [role="radiogroup"] label:nth-of-type(1)::before {{
  -webkit-mask-image:url("data:image/svg+xml,{_SEARCH_MASK}"); mask-image:url("data:image/svg+xml,{_SEARCH_MASK}");}}
[data-testid="stRadio"] [role="radiogroup"] label:nth-of-type(2)::before {{
  -webkit-mask-image:url("data:image/svg+xml,{_USERPLUS_MASK}"); mask-image:url("data:image/svg+xml,{_USERPLUS_MASK}");}}

/* ---- form section cards (st.container(border=True)) ---- */
[data-testid="stVerticalBlockBorderWrapper"] {{background:var(--bg-card) !important;
  border:1px solid var(--border) !important; border-radius:var(--r-lg) !important;
  padding:1.25rem 1.35rem 1.35rem !important; box-shadow:var(--shadow); margin-bottom:1.7rem;}}
.section-head {{display:flex; align-items:center; gap:10px; font-family:'Space Grotesk';
  font-weight:600; font-size:1.02rem; color:var(--text); margin:.1rem 0 1rem;}}
.section-head svg {{width:18px; height:18px; color:var(--accent-2);}}

/* ---- widget labels ---- */
[data-testid="stWidgetLabel"] label p, .stSlider label p {{color:var(--muted) !important;
  font-weight:500; font-size:.85rem;}}

/* ---- number inputs & selects ---- */
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
  background:var(--bg-elev) !important; border:1px solid var(--border) !important;
  border-radius:var(--r-sm) !important; overflow:hidden;}}
[data-testid="stNumberInput"] input {{background:transparent !important; color:var(--text) !important;}}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {{
  border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--accent-glow) !important;}}
[data-testid="stNumberInput"] input::-webkit-inner-spin-button,
[data-testid="stNumberInput"] input::-webkit-outer-spin-button {{-webkit-appearance:none; margin:0;}}
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{
  background:var(--bg-card) !important; color:var(--muted) !important; border:none !important;
  border-left:1px solid var(--border) !important; transition:background .15s ease, color .15s ease;}}
[data-testid="stNumberInputStepUp"]:hover, [data-testid="stNumberInputStepDown"]:hover {{
  background:var(--bg-elev) !important; color:var(--accent-2) !important;}}
[data-testid="stNumberInputStepUp"]:active, [data-testid="stNumberInputStepDown"]:active {{
  background:var(--accent) !important; color:#0B0D12 !important;}}
[data-testid="stNumberInput"] button svg {{width:15px; height:15px;}}

/* ---- sliders: thicker track, glowing thumb, floating value badge ---- */
.stSlider [data-baseweb="slider"] {{padding-top:16px;}}
.stSlider [data-baseweb="slider"] > div > div {{height:6px !important; border-radius:999px !important;}}
.stSlider [role="slider"] {{height:20px !important; width:20px !important; background:#fff !important;
  box-shadow:0 0 0 4px rgba(99,102,241,.20), 0 0 14px var(--accent-glow) !important;}}
[data-testid="stThumbValue"] {{background:var(--accent) !important; color:#0B0D12 !important;
  font-weight:600 !important; font-size:12px !important; padding:2px 9px; border-radius:999px;
  box-shadow:0 3px 10px var(--accent-glow);}}

/* ---- primary CTA (form submit) ---- */
[data-testid="stFormSubmitButton"] button, .stButton button {{
  background:linear-gradient(180deg,var(--accent-2),var(--accent)) !important; color:#0B0D12 !important;
  font-weight:600 !important; border:none !important; border-radius:var(--r) !important;
  box-shadow:0 6px 20px var(--accent-glow) !important;
  transition:transform .16s ease, box-shadow .16s ease !important;}}
[data-testid="stFormSubmitButton"] button {{width:100%; padding:.85rem 1rem !important; font-size:1rem !important;}}
[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover {{
  transform:translateY(-2px); box-shadow:0 12px 30px var(--accent-glow) !important; color:#0B0D12 !important;}}

/* ---- metrics (sidebar) ---- */
[data-testid="stMetric"] {{background:var(--bg-elev); border:1px solid var(--border);
  border-radius:var(--r); padding:14px 16px;}}
[data-testid="stMetricValue"] {{font-family:'Space Grotesk',sans-serif; font-weight:600;}}
[data-testid="stMetricLabel"] p {{color:var(--muted) !important;}}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{background:var(--bg-card); border-right:1px solid var(--border);}}
[data-testid="stSidebar"] h2 {{font-size:1.1rem; font-family:'Space Grotesk';}}

/* ---- expander ---- */
[data-testid="stExpander"] {{border:1px solid var(--border) !important; border-radius:var(--r) !important;
  background:var(--bg-card) !important; box-shadow:none !important; overflow:hidden; margin-top:1.4rem;}}
[data-testid="stExpander"] summary {{padding:14px 18px !important; color:var(--text) !important;
  font-weight:500;}}
[data-testid="stExpander"] summary:hover {{color:var(--accent-2) !important; background:var(--bg-elev) !important;}}

/* ---- result hero ---- */
.result {{position:relative; overflow:hidden; margin-top:1.6rem; padding:1.6rem 1.7rem;
  border-radius:var(--r-lg); border:1px solid var(--border);
  background:linear-gradient(180deg,var(--bg-card),rgba(20,24,33,.4));
  box-shadow:var(--shadow); animation:reveal .42s cubic-bezier(.22,1,.36,1) both;}}
.result.ok {{border-color:rgba(34,197,94,.28);}}
.result.bad {{border-color:rgba(244,63,94,.28);}}
.result::before {{content:""; position:absolute; top:0; left:0; right:0; height:3px;}}
.result.ok::before {{background:linear-gradient(90deg,var(--ok),transparent);}}
.result.bad::before {{background:linear-gradient(90deg,var(--bad),transparent);}}
.result-top {{display:flex; align-items:center; justify-content:space-between; gap:16px;}}
.risk-label {{color:var(--muted); font-size:.72rem; font-weight:600; letter-spacing:.09em;}}
.risk-value {{font-family:'Space Grotesk'; font-weight:700; font-size:3rem; line-height:1;}}
.risk-value span {{font-size:1.35rem; color:var(--muted); margin-left:2px;}}
.badge {{display:inline-flex; align-items:center; gap:8px; padding:10px 20px; border-radius:999px;
  font-weight:700; font-size:.95rem; letter-spacing:.02em;}}
.badge svg {{width:18px; height:18px;}}
.badge-ok {{background:rgba(34,197,94,.14); color:#4ADE80; border:1px solid rgba(34,197,94,.42);
  box-shadow:0 0 26px var(--ok-glow);}}
.badge-bad {{background:rgba(244,63,94,.14); color:#FB7185; border:1px solid rgba(244,63,94,.42);
  box-shadow:0 0 26px var(--bad-glow);}}
.gauge {{position:relative; height:10px; border-radius:999px; background:var(--bg-elev); margin:1.5rem 0 .4rem;}}
.gauge-fill {{position:absolute; left:0; top:0; height:100%; border-radius:999px;
  background:linear-gradient(90deg,var(--accent),var(--accent-2)); box-shadow:0 0 16px var(--accent-glow);
  animation:grow .6s ease both;}}
.gauge-mark {{position:absolute; top:-5px; width:3px; height:20px; background:#fff;
  border-radius:3px; transform:translateX(-1px);
  box-shadow:0 0 0 2px rgba(11,13,18,.65), 0 0 8px rgba(0,0,0,.5);}}
.gauge-mark::before {{content:""; position:absolute; top:-6px; left:50%; transform:translateX(-50%);
  width:8px; height:8px; border-radius:50%; background:#fff;
  box-shadow:0 0 0 2px rgba(11,13,18,.65), 0 0 6px rgba(0,0,0,.5);}}
.gauge-cap {{display:flex; justify-content:space-between; color:var(--muted); font-size:.72rem; margin-bottom:1.2rem;}}
.gauge-cap span:nth-child(2) {{color:var(--text); font-weight:600;}}
.verdict {{display:flex; align-items:flex-start; gap:10px; padding:13px 16px; border-radius:var(--r);
  font-size:.94rem; line-height:1.45; margin-bottom:1.4rem;}}
.verdict svg {{width:18px; height:18px; flex:0 0 18px; margin-top:1px;}}
.verdict b {{color:#fff;}}
.result.ok .verdict {{background:rgba(34,197,94,.09); border:1px solid rgba(34,197,94,.22); color:#BBF7D0;}}
.result.bad .verdict {{background:rgba(244,63,94,.09); border:1px solid rgba(244,63,94,.22); color:#FECDD3;}}
.reasons-h {{font-family:'Space Grotesk'; font-weight:600; font-size:1.05rem; margin-bottom:.15rem;}}
.reasons-sub {{color:var(--muted); font-size:.85rem; margin-bottom:.85rem;}}
.reason-row {{display:flex; align-items:center; gap:12px; padding:12px 14px; margin-bottom:8px;
  background:var(--bg-elev); border:1px solid var(--border); border-radius:var(--r);
  animation:reveal .45s ease both;
  transition:transform .15s ease, border-color .15s ease, box-shadow .15s ease, background-image .15s ease;}}
.reason-row:hover {{border-color:var(--border-strong); transform:translateY(-1px);
  background-image:linear-gradient(rgba(255,255,255,.03),rgba(255,255,255,.03));
  box-shadow:0 5px 16px rgba(0,0,0,.28);}}
.reason-ic {{display:flex; align-items:center; justify-content:center; width:30px; height:30px;
  flex:0 0 30px; border-radius:9px; background:rgba(99,102,241,.14); color:var(--accent-2);}}
.reason-ic svg {{width:16px; height:16px;}}
.reason-lbl {{color:var(--text); font-weight:500; font-size:.95rem;}}
.reason-lbl::first-letter {{text-transform:uppercase;}}

@keyframes reveal {{from {{opacity:0; transform:translateY(14px);}} to {{opacity:1; transform:none;}}}}
@keyframes grow {{from {{width:0 !important;}}}}

/* ---- responsiveness ---- */
@media (max-width:640px) {{
  .risk-value {{font-size:2.4rem;}}
  .result-top {{flex-direction:column; align-items:flex-start; gap:12px;}}
  .app-pill {{display:none;}}
  [data-testid="stRadio"] > div[role="radiogroup"] {{display:flex; width:100%;}}
  [data-testid="stRadio"] [role="radiogroup"] label {{flex:1; justify-content:center; padding:10px 12px;}}
}}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# cached, do-once loads

@st.cache_resource
def get_bundle():
    return scoring.load_bundle()


@st.cache_data
def get_pds():
    bundle = get_bundle()
    return scoring.predicted_pd(bundle["calibrated"], bundle["X"])


@st.cache_data
def get_medians():
    return scoring.training_medians()


bundle = get_bundle()
X, y = bundle["X"], bundle["y"]
pd_all = get_pds()


# sidebar: cost-ratio slider

st.sidebar.header("Cost settings")
st.sidebar.caption(
    "How many times more does a **missed default** cost than a **false decline**? "
    "Baseline (×1.0) uses the report's cost matrix, where the cutoff is 0.09."
)
cost_ratio = st.sidebar.slider(
    "Relative cost of a missed default", min_value=0.5, max_value=3.0,
    value=1.0, step=0.5,
)

c_fn, c_fp = scoring.build_costs(X, cost_ratio=cost_ratio)
t_star, approval = scoring.choose_threshold(y, pd_all, c_fn, c_fp)

st.sidebar.metric("Decision cutoff", f"{t_star:.0%} risk")
st.sidebar.metric("Approval rate", f"{approval:.1%}")
st.sidebar.caption(
    "Raise the cost and we decline more aggressively (lower cutoff, fewer "
    "approvals); lower it and the line relaxes."
)


# shared decision card

def render_decision(x_row):
    """Show the PD, approve/decline call, and plain-language reasons."""
    pd_value = float(scoring.predicted_pd(bundle["calibrated"], x_row)[0])
    decision = scoring.decide(pd_value, t_star)
    reasons = scoring.explain_applicant(bundle["raw"], x_row, bundle["features"], top_n=3)

    ok = decision == "APPROVE"
    fill = min(pd_value / 0.30, 1.0) * 100          # gauge fill, capped at 30%
    cutoff = min(t_star / 0.30, 1.0) * 100          # marker for the decision line
    badge_icon = ICON_CHECK if ok else ICON_X
    reason_icon = ICON_INFO if ok else ICON_ALERT
    if ok:
        verdict = ("This applicant's risk is <b>within our approval limit</b>, so the "
                   "loan is <b>approved</b>.")
        reason_sub = "The factors that most affected this applicant's risk score:"
    else:
        verdict = ("This applicant is <b>riskier than our approval cutoff</b>, so the "
                   "loan is <b>declined</b>.")
        reason_sub = "The factors that pushed this applicant's risk up the most:"

    rows = "".join(
        f'<div class="reason-row" style="animation-delay:{0.06 * i + 0.15:.2f}s">'
        f'<span class="reason-ic">{reason_icon}</span>'
        f'<span class="reason-lbl">{label}</span></div>'
        for i, (label, _) in enumerate(reasons)
    )

    html = (
        f'<div class="result {"ok" if ok else "bad"}">'
        '<div class="result-top">'
        '<div><div class="risk-label">ESTIMATED DEFAULT RISK</div>'
        f'<div class="risk-value">{pd_value * 100:.1f}<span>%</span></div></div>'
        f'<div class="badge {"badge-ok" if ok else "badge-bad"}">{badge_icon}'
        f'<span>{"APPROVE" if ok else "DECLINE"}</span></div>'
        '</div>'
        f'<div class="gauge"><div class="gauge-fill" style="width:{fill:.1f}%"></div>'
        f'<div class="gauge-mark" style="left:{cutoff:.1f}%"></div></div>'
        f'<div class="gauge-cap"><span>0%</span><span>cutoff {t_star:.0%}</span>'
        '<span>30%+</span></div>'
        f'<div class="verdict">{reason_icon}<span>{verdict}</span></div>'
        '<div class="reasons-h">Top reasons</div>'
        f'<div class="reasons-sub">{reason_sub}</div>'
        f'{rows}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    with st.expander("How this decision is made (technical detail)"):
        st.markdown(
            f"""
- This applicant's **probability of default is {pd_value:.1%}**.
- We **decline when the risk reaches {t_star:.0%}** (threshold t\\* = {t_star:.2f}) —
  the cutoff that **minimises expected loss**, not a naive 50%. A missed default
  costs far more than a false decline, so the line sits well below 50%.
- **Reason codes** are the top SHAP contributors — the same content a regulator
  expects on an adverse-action (decline) notice.
- Gender is **excluded** from the model by policy and used only for a fairness check.
            """
        )


# main panel

st.markdown(
    f'<div class="app-header">'
    f'<div class="app-logo">{ICON_LOGO}</div>'
    '<div class="app-heading"><div class="app-title">Credit-Risk Decision Engine</div>'
    '<div class="app-sub">Score default risk, decide by cost, explain every call — '
    'in plain language.</div></div>'
    '<div class="app-pill"><span class="dot">●</span> LIVE MODEL</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="seg-label">Mode</div>', unsafe_allow_html=True)
mode = st.radio(
    "Choose a mode", [MODE_LOOKUP, MODE_NEW],
    horizontal=True, label_visibility="collapsed",
)

if mode == MODE_LOOKUP:
    # mode 1: replay a held-out applicant
    applicant_id = st.selectbox(
        "Choose an applicant (SK_ID_CURR)", options=list(X.index),
        help="These are held-out validation applicants the model never trained on.",
    )
    render_decision(X.loc[[applicant_id]])

else:
    # mode 2: score a brand-new applicant
    st.caption(
        "Enter the handful of details that most affect the decision. Everything else "
        "is set to a typical (median) applicant automatically."
    )
    with st.form("new_applicant"):
        with st.container(border=True):
            st.markdown(f'<div class="section-head">{ICON_MONEY}<span>Loan &amp; income'
                        '</span></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            income = c1.number_input("Annual income", min_value=0, value=150000, step=5000)
            loan = c2.number_input("Loan amount", min_value=0, value=500000, step=5000)
            annuity = c1.number_input("Monthly payment", min_value=0, value=25000, step=1000)
            goods_price = c2.number_input("Price of item financed", min_value=0,
                                          value=450000, step=5000)

        with st.container(border=True):
            st.markdown(f'<div class="section-head">{ICON_USER}<span>About the applicant'
                        '</span></div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            age = c3.slider("Age", min_value=18, max_value=100, value=40)
            years_employed = c4.slider("Years employed", min_value=0, max_value=50, value=5)

        with st.container(border=True):
            st.markdown(f'<div class="section-head">{ICON_CARD}<span>Credit standing'
                        '</span></div>', unsafe_allow_html=True)
            ext_score = st.slider(
                "External credit-bureau score (0 = poor, 1 = excellent)",
                min_value=0.0, max_value=1.0, value=0.50, step=0.01,
            )
            missed = st.selectbox("Past missed-payment history",
                                  options=list(MISSED_PAYMENT_OPTIONS.keys()))
            card_usage = st.slider(
                "Credit-card usage level (0 = unused, 1 = maxed out)",
                min_value=0.0, max_value=1.0, value=0.25, step=0.01,
            )

        submitted = st.form_submit_button("Score applicant")

    if submitted:
        inputs = {
            "income": income, "loan": loan, "annuity": annuity,
            "goods_price": goods_price, "age": age, "years_employed": years_employed,
            "ext_score": ext_score,
            "missed_payment_rate": MISSED_PAYMENT_OPTIONS[missed],
            "card_usage": card_usage,
        }
        problems = scoring.validate_new_inputs(inputs)
        if problems:
            st.warning("Please fix these before scoring:")
            for p in problems:
                st.markdown(f"- {p}")
        else:
            x_row = scoring.build_new_row(inputs, get_medians())
            render_decision(x_row)

st.caption(
    "Educational portfolio project on the Home Credit dataset — not a real lending "
    "decision system."
)
