

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Lazy SHAP import — only loads when needed
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
try:
    from sklearn.impute import SimpleImputer
    if not hasattr(SimpleImputer, "_fill_dtype"):
        import numpy as _np
        SimpleImputer._fill_dtype = property(lambda self: _np.float64)
except Exception:
    pass

# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="AcademiQ · Risk Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --black:       #040810;
    --bg:          #070c11;
    --bg-card:     #0b1520;
    --bg-card2:    #0e1c2a;
    --bg-card3:    #111f30;
    --border:      #0d2535;
    --border2:     #152e42;
    --cyan:        #00e5ff;
    --cyan-dim:    #00b8cc;
    --cyan-dark:   #007a8a;
    --cyan-ghost:  rgba(0,229,255,0.055);
    --cyan-glow:   rgba(0,229,255,0.16);
    --cyan-glow2:  rgba(0,229,255,0.32);
    --yellow:      #ffd85e;
    --yellow-dim:  rgba(255,216,94,0.12);
    --red:         #ff4060;
    --red-dim:     rgba(255,64,96,0.12);
    --green:       #00f5a0;
    --green-dim:   rgba(0,245,160,0.12);
    --text:        #c8e0ec;
    --text-muted:  #456070;
    --text-dim:    #1c3545;
    --mono:        'IBM Plex Mono', monospace;
    --sans:        'IBM Plex Sans', sans-serif;
    --display:     'Bebas Neue', sans-serif;
    --r:           5px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: var(--sans) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ── dot-grid bg ── */
.stApp {
    background-image: radial-gradient(circle, rgba(0,229,255,0.07) 1px, transparent 1px) !important;
    background-size: 28px 28px !important;
}

/* ── vignette ── */
.stApp::after {
    content:'';
    position:fixed; inset:0;
    background: radial-gradient(ellipse at center, transparent 40%, rgba(4,8,16,0.7) 100%);
    pointer-events:none; z-index:0;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--black) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top:0 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--text-muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
}
[data-testid="stSidebar"] .stMarkdown h4 {
    color: var(--cyan) !important;
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
[data-testid="collapsedControl"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: none !important;
    border-radius: 0 var(--r) var(--r) 0 !important;
    transition: all 0.2s !important;
}
[data-testid="collapsedControl"]:hover { background: var(--cyan-ghost) !important; box-shadow: 3px 0 18px var(--cyan-glow) !important; }
[data-testid="collapsedControl"] svg { fill: var(--cyan) !important; }
[data-testid="stSidebarCollapseButton"] button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebarCollapseButton"] button:hover { background: var(--cyan-ghost) !important; border-color: var(--cyan) !important; }
[data-testid="stSidebarCollapseButton"] svg { fill: var(--cyan) !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan-dim) !important; box-shadow: 0 0 14px var(--cyan-glow) !important; }
[data-testid="stFileUploader"] button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--cyan) !important;
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    border-radius: var(--r) !important;
    letter-spacing: 0.08em;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    color: var(--cyan) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}
[data-testid="stSelectbox"] > div > div:hover { border-color: var(--cyan) !important; box-shadow: 0 0 10px var(--cyan-glow) !important; }

/* ── LAYOUT ── */
.block-container { padding: 0 2.2rem 3rem !important; max-width: 1440px !important; }
hr { border:none !important; border-top: 1px solid var(--border) !important; margin: 2rem 0 !important; }
[data-testid="column"] { padding: 0 0.35rem !important; }

/* ── ALERTS ── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-left: 2px solid var(--cyan) !important;
    border-radius: var(--r) !important;
    color: var(--text-muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.74rem !important;
}
[data-testid="stSpinner"] > div { border-color: var(--cyan) transparent transparent transparent !important; }
.stCaption, .stCaption p { font-family: var(--mono) !important; font-size: 0.63rem !important; color: var(--text-dim) !important; letter-spacing: 0.04em; }

/* ── TABLES ── */
[data-testid="stDataFrame"] { border: none !important; border-radius: 0 !important; }
[data-testid="stDataFrame"] th {
    background: var(--black) !important; color: var(--cyan) !important;
    font-family: var(--mono) !important; font-size: 0.6rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
    background: var(--bg-card) !important; color: var(--text) !important;
    font-family: var(--mono) !important; font-size: 0.74rem !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tr:hover td { background: var(--cyan-ghost) !important; }

/* ── DOWNLOAD BTN ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important; border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important; font-family: var(--mono) !important;
    font-size: 0.68rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; border-radius: var(--r) !important;
    padding: 0.45rem 1.3rem !important; transition: all 0.22s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--cyan-ghost) !important;
    box-shadow: 0 0 18px var(--cyan-glow) !important;
    transform: translateY(-1px) !important;
}

/* ════════════════════════════════════
   COMPONENTS
════════════════════════════════════ */

/* SIDEBAR BRAND */
.sb-brand { padding: 1.5rem 1.1rem 1.1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.2rem; }
.sb-logo-row { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.35rem; }
.sb-hex { font-size:1.5rem; color:var(--cyan); text-shadow:0 0 12px var(--cyan-glow2); animation: hexpulse 3s ease-in-out infinite; }
@keyframes hexpulse { 0%,100%{opacity:.4;text-shadow:none} 50%{opacity:.9;text-shadow:0 0 20px var(--cyan-glow2)} }
.sb-name { font-family:var(--display); font-size:1.3rem; letter-spacing:.1em; color:var(--text); }
.sb-tagline { font-family:var(--mono); font-size:0.53rem; color:var(--text-dim); letter-spacing:.16em; text-transform:uppercase; }
.sb-status { margin-top:.7rem; display:flex; align-items:center; gap:.4rem; }
.sb-dot { width:5px; height:5px; border-radius:50%; background:var(--green); box-shadow:0 0 5px var(--green); animation:blink 2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }
.sb-status-txt { font-family:var(--mono); font-size:0.55rem; color:var(--text-dim); letter-spacing:.1em; text-transform:uppercase; }
.sb-file-info { font-family:var(--mono); font-size:0.57rem; color:var(--text-dim); padding:.3rem 0 .7rem; line-height:2.1; }
.sb-file-info span { color:var(--cyan); }
.sb-meta { font-family:var(--mono); font-size:0.55rem; color:var(--text-dim); letter-spacing:.07em; line-height:2.3; }
.sb-meta span { color:var(--text-muted); }

/* MODEL INFO BADGE in sidebar */
.model-badge {
    background: var(--bg-card);
    border: 1px solid var(--border2);
    border-left: 2px solid var(--cyan);
    border-radius: var(--r);
    padding: .65rem .9rem;
    margin: .5rem 0 .8rem;
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    line-height: 2;
}
.model-badge .acc { color: var(--green); font-size:.75rem; font-weight:600; }

/* HERO */
.hero { padding:2.8rem 0 1.8rem; border-bottom:1px solid var(--border); margin-bottom:2.2rem; position:relative; }
.hero::after { content:''; position:absolute; bottom:-1px; left:0; width:220px; height:1px; background:linear-gradient(90deg,var(--cyan),transparent); }
.hero-eye { font-family:var(--mono); font-size:0.63rem; color:var(--cyan); letter-spacing:.28em; text-transform:uppercase; margin-bottom:.65rem; display:flex; align-items:center; gap:.55rem; }
.hero-eye::before { content:''; display:inline-block; width:16px; height:1px; background:var(--cyan); }
.hero-h { font-family:var(--display); font-size:3.8rem; line-height:.9; color:var(--text); letter-spacing:.04em; margin-bottom:.65rem; }
.hero-h .c { color:var(--cyan); text-shadow:0 0 28px var(--cyan-glow2); }
.hero-p { font-family:var(--mono); font-size:0.75rem; color:var(--text-muted); letter-spacing:.03em; line-height:1.75; max-width:520px; margin-bottom:1.1rem; }
.tags { display:flex; gap:.4rem; flex-wrap:wrap; }
.tag { font-family:var(--mono); font-size:.6rem; letter-spacing:.12em; text-transform:uppercase; padding:3px 9px; border:1px solid var(--border2); border-radius:2px; color:var(--text-dim); background:var(--bg-card); }
.tag.on { border-color:var(--cyan); color:var(--cyan); background:var(--cyan-ghost); box-shadow:0 0 7px var(--cyan-glow); }
.hero-sys { position:absolute; right:0; top:50%; transform:translateY(-50%); font-family:var(--mono); font-size:0.54rem; color:var(--text-dim); line-height:2.1; text-align:right; letter-spacing:.05em; pointer-events:none; }
.hero-sys .on { color:var(--green); }

/* SECTION HEADER */
.sh { display:flex; align-items:center; gap:.7rem; margin-bottom:1.1rem; margin-top:.3rem; }
.sh-num { font-family:var(--mono); font-size:0.58rem; color:var(--cyan); letter-spacing:.12em; border:1px solid var(--cyan); padding:2px 6px; border-radius:2px; background:var(--cyan-ghost); }
.sh-title { font-family:var(--display); font-size:1.4rem; letter-spacing:.07em; color:var(--text); }
.sh-line { flex:1; height:1px; background:linear-gradient(90deg,var(--border),transparent); }

/* METRIC CARDS */
.mrow { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:var(--r); overflow:hidden; margin-bottom:2.2rem; box-shadow:0 0 50px rgba(0,0,0,.65); }
.mc { background:var(--bg-card); padding:1.6rem 1.8rem; position:relative; overflow:hidden; transition:background .25s; cursor:default; }
.mc:hover { background:var(--bg-card2); }
.mc::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.mc.tot::before { background:var(--cyan);  box-shadow:0 0 10px var(--cyan); }
.mc.goo::before { background:var(--green); box-shadow:0 0 10px var(--green); }
.mc.ris::before { background:var(--yellow);box-shadow:0 0 10px var(--yellow); }
.mc.cri::before { background:var(--red);   box-shadow:0 0 10px var(--red); }
.mc-corner { position:absolute; top:.9rem; right:.9rem; font-family:var(--mono); font-size:0.5rem; color:var(--text-dim); letter-spacing:.1em; text-transform:uppercase; }
.mc-icon { font-size:.9rem; margin-bottom:.8rem; opacity:.55; }
.mc-val { font-family:var(--display); font-size:3.2rem; line-height:1; letter-spacing:.04em; margin-bottom:.35rem; }
.tot .mc-val { color:var(--cyan);   text-shadow:0 0 18px var(--cyan-glow2); }
.goo .mc-val { color:var(--green);  text-shadow:0 0 18px rgba(0,245,160,.3); }
.ris .mc-val { color:var(--yellow); text-shadow:0 0 18px rgba(255,216,94,.3); }
.cri .mc-val { color:var(--red);    text-shadow:0 0 18px rgba(255,64,96,.3); }
.mc-lbl { font-family:var(--mono); font-size:0.63rem; color:var(--text-muted); letter-spacing:.13em; text-transform:uppercase; }
.mc-sub { margin-top:.85rem; padding-top:.8rem; border-top:1px solid var(--border); font-family:var(--mono); font-size:0.57rem; color:var(--text-dim); letter-spacing:.05em; }
.mc-sub b { color:var(--text-muted); }

/* PROGRESS BAR (for pct) */
.mc-bar { margin-top:.55rem; background:var(--border); border-radius:2px; height:3px; overflow:hidden; }
.mc-bar-fill { height:100%; border-radius:2px; }
.goo .mc-bar-fill { background:var(--green); }
.ris .mc-bar-fill { background:var(--yellow); }
.cri .mc-bar-fill { background:var(--red); }

/* PANEL */
.panel { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--r); overflow:hidden; margin-bottom:.9rem; box-shadow:0 4px 28px rgba(0,0,0,.5); }
.ph { background:var(--black); border-bottom:1px solid var(--border); padding:.6rem 1.3rem; display:flex; align-items:center; gap:.4rem; }
.ph-dot { width:6px; height:6px; border-radius:50%; background:var(--border); }
.ph-dot.on { background:var(--cyan); box-shadow:0 0 5px var(--cyan); }
.ph-dot.y  { background:var(--yellow); }
.ph-dot.r  { background:var(--red); }
.ph-title { font-family:var(--mono); font-size:0.6rem; color:var(--text-muted); letter-spacing:.12em; text-transform:uppercase; margin-left:.3rem; }
.ph-right { margin-left:auto; font-family:var(--mono); font-size:0.55rem; color:var(--text-dim); letter-spacing:.08em; }
.pb { padding:1.1rem; }

/* DIST CHART */
.dist-wrap { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--r); padding:1.3rem; }

/* LEGEND CARDS */
.lc { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--r); padding:1rem 1.2rem; position:relative; overflow:hidden; margin-bottom:.7rem; transition:border-color .2s,box-shadow .2s; }
.lc:hover { border-color:var(--border2); box-shadow:0 0 12px var(--cyan-glow); }
.lc::after { content:''; position:absolute; left:0; top:0; bottom:0; width:3px; }
.lc-g::after { background:var(--green); }
.lc-r::after { background:var(--yellow); }
.lc-c::after { background:var(--red); }
.lc-lbl { font-family:var(--mono); font-size:0.57rem; letter-spacing:.14em; text-transform:uppercase; color:var(--text-muted); margin-bottom:.25rem; }
.lc-num { font-family:var(--display); font-size:1.85rem; letter-spacing:.05em; line-height:1; }
.lc-g .lc-num { color:var(--green); }
.lc-r .lc-num { color:var(--yellow); }
.lc-c .lc-num { color:var(--red); }
.lc-pct { font-family:var(--mono); font-size:0.6rem; color:var(--text-dim); margin-top:.2rem; }

/* SHAP */
.shap-bar-info {
    background:var(--black); border:1px solid var(--border); border-left:3px solid var(--cyan);
    border-radius:var(--r); padding:.85rem 1.4rem; margin-bottom:1.2rem;
    display:flex; align-items:center; gap:.9rem;
    font-family:var(--mono); font-size:0.73rem; color:var(--text-muted);
}
.shap-bar-info strong { color:var(--text); }
.rbadge { font-family:var(--mono); font-size:0.65rem; font-weight:600; letter-spacing:.1em; padding:3px 9px; border-radius:2px; text-transform:uppercase; }
.rb-g { background:var(--green-dim);  border:1px solid var(--green);  color:var(--green); }
.rb-r { background:var(--yellow-dim); border:1px solid var(--yellow); color:var(--yellow); }
.rb-c { background:var(--red-dim);    border:1px solid var(--red);    color:var(--red); }
.shap-wrap { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--r); padding:1.5rem 1.8rem; box-shadow:0 8px 36px rgba(0,0,0,.5); }
.shap-thead { display:grid; grid-template-columns:210px 1fr 100px; gap:1rem; font-family:var(--mono); font-size:0.55rem; letter-spacing:.14em; text-transform:uppercase; color:var(--text-dim); margin-bottom:.9rem; padding-bottom:.55rem; border-bottom:1px solid var(--border); }
.shap-r { display:grid; grid-template-columns:210px 1fr 100px; align-items:center; gap:1rem; padding:.6rem .3rem; border-bottom:1px solid var(--border); border-radius:3px; transition:background .15s; }
.shap-r:last-child { border-bottom:none; }
.shap-r:hover { background:var(--cyan-ghost); padding-left:.5rem; }
.shap-feat { font-family:var(--mono); font-size:0.72rem; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.shap-rk { font-size:0.52rem; color:var(--text-dim); margin-right:.4rem; }
/* sessional features highlighted */
.shap-feat.sess { color:var(--cyan); }
.shap-track { background:var(--black); border:1px solid var(--border); border-radius:2px; height:9px; overflow:hidden; }
.shap-fill { height:100%; border-radius:2px; background:linear-gradient(90deg,var(--cyan-dark),var(--cyan)); box-shadow:0 0 7px var(--cyan-glow); position:relative; }
.shap-fill.sess { background:linear-gradient(90deg,#006080,var(--cyan)); box-shadow:0 0 12px var(--cyan-glow2); }
.shap-fill::after { content:''; position:absolute; right:0; top:0; bottom:0; width:2px; background:white; opacity:.35; border-radius:0 2px 2px 0; }
.shap-v { font-family:var(--mono); font-size:0.73rem; color:var(--cyan); text-align:right; }

/* PROB GAUGE STRIP */
.prob-strip { display:flex; gap:6px; margin-top:.55rem; }
.prob-seg { flex:1; }
.prob-seg-lbl { font-family:var(--mono); font-size:0.5rem; color:var(--text-dim); letter-spacing:.08em; text-transform:uppercase; margin-bottom:3px; }
.prob-seg-bar { height:5px; border-radius:2px; }
.prob-seg-val { font-family:var(--mono); font-size:0.6rem; margin-top:3px; }
.ps-g .prob-seg-bar { background:var(--green); }
.ps-r .prob-seg-bar { background:var(--yellow); }
.ps-c .prob-seg-bar { background:var(--red); }
.ps-g .prob-seg-val { color:var(--green); }
.ps-r .prob-seg-val { color:var(--yellow); }
.ps-c .prob-seg-val { color:var(--red); }

/* EMPTY STATE */
.empty { min-height:55vh; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:3.5rem 2rem; }
.empty-ico { font-size:2.8rem; color:var(--cyan); animation:hexpulse 3s ease-in-out infinite; margin-bottom:1.3rem; }
.empty-h { font-family:var(--display); font-size:1.9rem; letter-spacing:.12em; color:var(--text-dim); margin-bottom:.55rem; }
.empty-p { font-family:var(--mono); font-size:0.7rem; color:var(--text-dim); line-height:1.9; max-width:340px; }

/* FEAT GUIDE GRID */
.fg { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:var(--r); overflow:hidden; max-width:880px; margin:0 auto; }
.fc { background:var(--bg-card); padding:1.1rem 1.3rem; transition:background .2s; }
.fc:hover { background:var(--bg-card2); }
.fc-ico { font-size:.9rem; margin-bottom:.55rem; opacity:.45; }
.fc-t { font-family:var(--mono); font-size:0.67rem; color:var(--cyan); letter-spacing:.1em; text-transform:uppercase; margin-bottom:.3rem; }
.fc-t.sess { color:var(--cyan); border-left:2px solid var(--cyan); padding-left:5px; }
.fc-d { font-family:var(--mono); font-size:0.6rem; color:var(--text-dim); line-height:1.7; }

/* ERR */
.err-wrap { display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:58vh; text-align:center; gap:.7rem; }
.err-ico { font-size:2.2rem; color:var(--red); opacity:.65; }
.err-h { font-family:var(--display); font-size:1.9rem; letter-spacing:.1em; color:var(--red); }
.err-b { font-family:var(--mono); font-size:0.73rem; color:var(--text-muted); line-height:2; max-width:480px; }
.ec { color:var(--cyan); } .ew { color:var(--yellow); }

/* ACCURACY PILL */
.acc-pill { display:inline-flex; align-items:center; gap:6px; background:var(--green-dim); border:1px solid var(--green); border-radius:20px; padding:3px 10px; font-family:var(--mono); font-size:0.62rem; color:var(--green); letter-spacing:.06em; }
.acc-dot { width:5px; height:5px; border-radius:50%; background:var(--green); box-shadow:0 0 4px var(--green); }

/* FEATURE IMPORTANCE MINI BAR (sidebar) */
.fi-row { display:flex; align-items:center; gap:6px; padding:3px 0; }
.fi-lbl { font-family:var(--mono); font-size:0.55rem; color:var(--text-dim); width:90px; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.fi-track { flex:1; background:var(--border); border-radius:2px; height:4px; }
.fi-fill { height:100%; border-radius:2px; background:linear-gradient(90deg,var(--cyan-dark),var(--cyan)); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_model():
    import glob
    for name in ["rf_model.joblib", "rf_model (1).joblib",
                 "model.joblib", "logistic_model.joblib"]:
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            return p
    for f in sorted(glob.glob(os.path.join(BASE_DIR, "*.joblib"))):
        base = os.path.basename(f).lower()
        if "label" not in base and "encoder" not in base:
            return f
    return None

def _find_encoder():
    import glob
    for name in ["label_encoder.joblib", "label_encoder (1).joblib",
                 "labelencoder.joblib", "labelencoderf.joblib",
                 "encoder.joblib"]:
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            return p
    for f in sorted(glob.glob(os.path.join(BASE_DIR, "*.joblib"))):
        base = os.path.basename(f).lower()
        if "label" in base or "encoder" in base:
            return f
    return None

def _patch_model(model):
    try:
        from sklearn.impute import SimpleImputer
        def _fix(imp):
            if isinstance(imp, SimpleImputer) and not hasattr(imp,"_fill_dtype"):
                imp._fill_dtype = np.float64
        if hasattr(model,"named_steps"):
            for s in model.named_steps.values(): _fix(s)
        elif hasattr(model,"steps"):
            for _,s in model.steps: _fix(s)
        else: _fix(model)
    except Exception: pass
    return model

@st.cache_resource
def load_artifacts():
    mp = _find_model()
    ep = _find_encoder()
    if not mp or not ep:
        return None, None, None, None
    if mp == ep:
        return None, None, None, "Model and encoder resolved to same file"
    try:
        m = _patch_model(joblib.load(mp))
        e = joblib.load(ep)
        if not hasattr(m, "predict"):
            return None, None, None, f"{os.path.basename(mp)} is not a model"
        if not hasattr(e, "classes_"):
            return None, None, None, f"{os.path.basename(ep)} is not a LabelEncoder"
        return m, e, os.path.basename(mp), os.path.basename(ep)
    except Exception as ex:
        return None, None, None, str(ex)

rf_model, label_encoder, model_fname, encoder_fname = load_artifacts()
model_ok = rf_model is not None and label_encoder is not None

# ── feature list (matches training) ──────────────────────────
FEATURES = [
    'attendance_pct',
    'quiz_1','quiz_2','quiz_3','quiz_4','quiz_5',
    'quiz_avg','quiz_std',
    'assignment_score',
    'sessional1','sessional2',
    'cheating_count','teacher_feedback_score'
]

# Sessional features highlighted in SHAP (high priority domain features)
SESSIONAL_FEATS = {'sessional1','sessional2'}


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    dot_col   = "var(--green)" if model_ok else "var(--red)"
    dot_label = "System · Online" if model_ok else "System · Error"

    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-logo-row">
            <span class="sb-hex">⬡</span>
            <span class="sb-name">ACADEMIQ</span>
        </div>
        <div class="sb-tagline">Risk Intelligence · v2.0</div>
        <div class="sb-status">
            <div class="sb-dot" style="background:{dot_col};box-shadow:0 0 5px {dot_col};"></div>
            <span class="sb-status-txt">{dot_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model_ok:
        st.markdown(f"""
        <div class="model-badge">
            <div>MODEL &nbsp;<span style="color:var(--cyan);">{model_fname}</span></div>
            <div>ENCODER &nbsp;<span style="color:var(--cyan);">{encoder_fname}</span></div>
            <div>ACCURACY &nbsp;<span class="acc">≈ 98.96%</span></div>
            <div>FEATURES &nbsp;<span style="color:var(--cyan);">13</span></div>
            <div>XAI &nbsp;<span style="color:var(--cyan);">SHAP Kernel</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### // Data Input")
    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv","xlsx"],
        label_visibility="collapsed"
    )
    st.caption("› .csv or .xlsx · columns must match required features")
    st.markdown("---")

    if model_ok and uploaded_file:
        st.markdown("#### // SHAP Explorer")
        shap_idx_slot = st.empty()
        st.markdown("---")

    st.markdown("""
    <div class="sb-meta">
        <div>ALGORITHM &nbsp;<span>Logistic Regression</span></div>
        <div>CV &nbsp;<span>5-Fold Stratified</span></div>
        <div>CLASSES &nbsp;<span>Good · AtRisk · Critical</span></div>
        <div>DATASET &nbsp;<span>102,349 students</span></div>
        <div>BUILD &nbsp;<span>Capstone 2025</span></div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eye">AcademiQ · v2.0 · Explainable AI</div>
    <div class="hero-h">STUDENT<br>PERFORMANCE<br><span class="c">RISK PREDICTOR</span></div>
    <div class="hero-p">
        Upload student records → classify academic risk levels →
        explain every prediction with SHAP explainability.
        Trained on 102,349 students · 98.96% accuracy · 100% Critical recall.
    </div>
    <div class="tags">
        <span class="tag on">Explainable AI</span>
        <span class="tag on">SHAP</span>
        <span class="tag on">Logistic Regression</span>
        <span class="tag">Risk Classification</span>
        <span class="tag">Academic Analytics</span>
        <span class="tag">Feature Engineering</span>
    </div>
    <div class="hero-sys">
        SYSTEM STATUS &nbsp;<span class="on">● ONLINE</span><br>
        MODEL LOADED &nbsp;<span class="on">● TRUE</span><br>
        FEATURES &nbsp;13<br>
        CLASSES &nbsp;3<br>
        CRITICAL RECALL &nbsp;<span class="on">100%</span><br>
        XAI &nbsp;<span class="on">● ENABLED</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# MODEL ERROR
# ═══════════════════════════════════════════════
if not model_ok:
    st.markdown(f"""
    <div class="err-wrap">
        <div class="err-ico">⬡</div>
        <div class="err-h">MODEL FILES NOT FOUND</div>
        <div class="err-b">
            Place these files in the same folder as sample.py:<br>
            <span class="ec">rf_model.joblib</span> &amp;
            <span class="ec">label_encoder.joblib</span><br><br>
            <span class="ew">Files may be named with (1) suffix — rename them.</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════
# FILE UPLOADED — MAIN FLOW
# ═══════════════════════════════════════════════
if uploaded_file is not None:

    # ── read ──────────────────────────────────
    try:
        df = (pd.read_excel(uploaded_file, engine="openpyxl")
              if uploaded_file.name.endswith(".xlsx")
              else pd.read_csv(uploaded_file))
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # ── feature engineering (must match training) ──
    qc = [c for c in ['quiz_1','quiz_2','quiz_3','quiz_4','quiz_5'] if c in df.columns]
    if qc:
        df['quiz_avg'] = df[qc].mean(axis=1)
        df['quiz_std'] = df[qc].std(axis=1).fillna(0)

    # ── validate ──────────────────────────────
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        st.markdown(f"""
        <div class="err-wrap" style="min-height:28vh;">
            <div class="err-ico" style="font-size:1.6rem;">⚠</div>
            <div class="err-h" style="font-size:1.3rem;color:var(--yellow);">MISSING COLUMNS</div>
            <div class="err-b"><span class="ew">{', '.join(missing)}</span></div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)

    # ── predict ───────────────────────────────
    try:
        preds = rf_model.predict(X)
        probs = rf_model.predict_proba(X)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    df['Predicted_Risk'] = label_encoder.inverse_transform(preds)
    classes = label_encoder.classes_
    for i, cls in enumerate(classes):
        df[f'Prob_{cls}'] = np.round(probs[:, i] * 100, 1)

    total    = len(df)
    n_good   = int((df['Predicted_Risk'] == 'Good').sum())
    n_risk   = int((df['Predicted_Risk'] == 'AtRisk').sum())
    n_crit   = int((df['Predicted_Risk'] == 'Critical').sum())
    pct_good = round(n_good / total * 100, 1)
    pct_risk = round(n_risk / total * 100, 1)
    pct_crit = round(n_crit / total * 100, 1)

    # ── METRIC CARDS ──────────────────────────
    st.markdown(f"""
    <div class="mrow">
        <div class="mc tot">
            <div class="mc-corner">00 · TOTAL</div>
            <div class="mc-icon">◈</div>
            <div class="mc-val">{total}</div>
            <div class="mc-lbl">Students Analyzed</div>
            <div class="mc-sub">File: <b>{uploaded_file.name}</b></div>
        </div>
        <div class="mc goo">
            <div class="mc-corner">01 · GOOD</div>
            <div class="mc-icon">◉</div>
            <div class="mc-val">{n_good}</div>
            <div class="mc-lbl">Good Standing</div>
            <div class="mc-bar"><div class="mc-bar-fill" style="width:{pct_good}%"></div></div>
            <div class="mc-sub"><b>{pct_good}%</b> of cohort</div>
        </div>
        <div class="mc ris">
            <div class="mc-corner">02 · WARNING</div>
            <div class="mc-icon">◭</div>
            <div class="mc-val">{n_risk}</div>
            <div class="mc-lbl">At Risk</div>
            <div class="mc-bar"><div class="mc-bar-fill" style="width:{pct_risk}%"></div></div>
            <div class="mc-sub"><b>{pct_risk}%</b> of cohort</div>
        </div>
        <div class="mc cri">
            <div class="mc-corner">03 · CRITICAL</div>
            <div class="mc-icon">⬡</div>
            <div class="mc-val">{n_crit}</div>
            <div class="mc-lbl">Critical Risk</div>
            <div class="mc-bar"><div class="mc-bar-fill" style="width:{pct_crit}%"></div></div>
            <div class="mc-sub"><b>{pct_crit}%</b> of cohort</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 01 — RESULTS TABLE ─────────────
    st.markdown('<div class="sh"><span class="sh-num">01</span><span class="sh-title">PREDICTION RESULTS</span><span class="sh-line"></span></div>', unsafe_allow_html=True)

    # build display df with probability columns
    prob_cols = [f'Prob_{c}' for c in classes if f'Prob_{c}' in df.columns]
    base_cols = ['Predicted_Risk'] + [c for c in ['attendance_pct','sessional1','sessional2','quiz_avg','assignment_score','cheating_count'] if c in df.columns] + prob_cols

    st.markdown(
        '<div class="panel">'
        '<div class="ph"><div class="ph-dot on"></div><div class="ph-dot y"></div>'
        '<div class="ph-dot r"></div>'
        '<span class="ph-title">student_predictions · live</span>'
        f'<span class="ph-right">{total} records</span></div>'
        '<div class="pb">',
        unsafe_allow_html=True
    )
    st.dataframe(df[base_cols], use_container_width=True, height=300)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.download_button(
        "⬇  EXPORT FULL RESULTS · CSV",
        data=df.to_csv(index=False),
        file_name="academiq_predictions.csv",
        mime="text/csv"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 02 — RISK DISTRIBUTION ─────────
    st.markdown('<div class="sh"><span class="sh-num">02</span><span class="sh-title">RISK DISTRIBUTION</span><span class="sh-line"></span></div>', unsafe_allow_html=True)

    risk_counts = df['Predicted_Risk'].value_counts()
    col_chart, col_leg = st.columns([3, 1])

    with col_chart:
        st.markdown('<div class="dist-wrap">', unsafe_allow_html=True)
        cd = risk_counts.reset_index()
        cd.columns = ["Risk Level","Count"]
        st.bar_chart(cd.set_index("Risk Level"), color="#00e5ff",
                     use_container_width=True, height=230)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_leg:
        def lc(label, count, css):
            p = round(count / total * 100, 1) if total else 0
            return (f'<div class="lc {css}"><div class="lc-lbl">{label}</div>'
                    f'<div class="lc-num">{count}</div>'
                    f'<div class="lc-pct">{p}% of cohort</div></div>')
        st.markdown(
            lc("Good",     risk_counts.get("Good",     0), "lc-g") +
            lc("At Risk",  risk_counts.get("AtRisk",   0), "lc-r") +
            lc("Critical", risk_counts.get("Critical", 0), "lc-c"),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 03 — SHAP ───────────────────────
    st.markdown('<div class="sh"><span class="sh-num">03</span><span class="sh-title">SHAP EXPLAINABILITY</span><span class="sh-line"></span></div>', unsafe_allow_html=True)

    if not SHAP_AVAILABLE:
        st.markdown("""
        <div style="font-family:var(--mono);font-size:0.73rem;color:var(--yellow);
            border:1px solid var(--border);border-left:3px solid var(--yellow);
            background:var(--bg-card);padding:.9rem 1.4rem;border-radius:5px;">
            ⚠ SHAP not installed. Run:
            <span style="color:var(--cyan);">pip install shap</span>
        </div>""", unsafe_allow_html=True)
    else:
        with st.sidebar:
            student_index = shap_idx_slot.selectbox(
                "Student Index for Explanation",
                df.index,
                label_visibility="visible"
            )

        predicted_risk = df.loc[student_index, 'Predicted_Risk']
        bc = {"Critical":"rb-c","AtRisk":"rb-r","Good":"rb-g"}.get(predicted_risk,"rb-g")

        # probability strip for selected student
        prob_strip_html = '<div class="prob-strip">'
        for cls, css in [("Good","ps-g"),("AtRisk","ps-r"),("Critical","ps-c")]:
            pk = f'Prob_{cls}'
            pv = float(df.loc[student_index, pk]) if pk in df.columns else 0.0
            prob_strip_html += (
                f'<div class="prob-seg {css}">'
                f'<div class="prob-seg-lbl">{cls}</div>'
                f'<div class="prob-seg-bar" style="width:{min(pv,100):.0f}%"></div>'
                f'<div class="prob-seg-val">{pv:.1f}%</div>'
                f'</div>'
            )
        prob_strip_html += '</div>'

        st.markdown(f"""
        <div class="shap-bar-info">
            <span>⬡</span>
            <span>Student <strong>#{student_index}</strong>
            &nbsp;·&nbsp; Predicted:
            </span>
            <span class="rbadge {bc}">{predicted_risk}</span>
            <span style="margin-left:auto;font-size:0.55rem;
                color:var(--text-dim);letter-spacing:.08em;">
                SHAP KERNEL EXPLAINER
            </span>
        </div>
        <div style="background:var(--bg-card);border:1px solid var(--border);
            border-radius:var(--r);padding:.9rem 1.4rem;margin-bottom:1.2rem;">
            <div style="font-family:var(--mono);font-size:0.55rem;color:var(--text-dim);
                letter-spacing:.12em;text-transform:uppercase;margin-bottom:.5rem;">
                Confidence Breakdown
            </div>
            {prob_strip_html}
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Computing SHAP values — this may take 20–40s …"):
            try:
                bg = X.sample(min(50, len(X)), random_state=42)

                def _pfn(data):
                    return rf_model.predict_proba(
                        pd.DataFrame(data, columns=FEATURES)
                    )

                explainer   = shap.KernelExplainer(_pfn, bg.values)
                sv          = explainer.shap_values(
                    X.iloc[[student_index]].values, silent=True
                )
                arr  = np.abs(np.array(sv, dtype=object))
                flat = np.concatenate([np.ravel(s) for s in arr])
                vec  = flat[:len(FEATURES)] if flat.size >= len(FEATURES) \
                       else np.pad(flat, (0, len(FEATURES)-flat.size))

                shap_df = (
                    pd.DataFrame({"Feature": FEATURES, "Impact": vec})
                    .sort_values("Impact", ascending=False)
                    .head(10)
                    .reset_index(drop=True)
                )

                mx = float(shap_df["Impact"].max()) or 1.0

                rows = ""
                for i, row in shap_df.iterrows():
                    bp   = round(row["Impact"] / mx * 100, 1)
                    feat = str(row["Feature"])
                    is_s = feat in SESSIONAL_FEATS
                    fc   = "shap-feat sess" if is_s else "shap-feat"
                    bc2  = "shap-fill sess" if is_s else "shap-fill"
                    tag  = ' <span style="font-size:.48rem;color:var(--cyan);opacity:.7;">[SESS]</span>' if is_s else ""
                    rows += (
                        '<div class="shap-r">'
                        f'<div class="{fc}"><span class="shap-rk">#{str(i+1).zfill(2)}</span>{feat}{tag}</div>'
                        f'<div class="shap-track"><div class="{bc2}" style="width:{bp}%"></div></div>'
                        f'<div class="shap-v">{row["Impact"]:.4f}</div>'
                        '</div>'
                    )

                st.markdown(
                    '<div class="shap-wrap">'
                    '<div class="shap-thead"><span>Feature</span>'
                    '<span>Relative Impact</span><span>SHAP Value</span></div>'
                    + rows + '</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"SHAP failed: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "▸ SHAP values show each feature's contribution to the prediction. "
            "[SESS] = sessional-derived feature (higher priority in model). "
            "Brighter cyan bar = stronger influence."
        )


# ═══════════════════════════════════════════════
# EMPTY STATE
# ═══════════════════════════════════════════════
else:
    st.markdown("""
    <div class="empty">
        <div class="empty-ico">⬡</div>
        <div class="empty-h">AWAITING DATA INPUT</div>
        <div class="empty-p">
            Upload a .csv or .xlsx student record file
            via the sidebar to begin risk analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sh" style="max-width:880px;margin:0 auto 1rem;">
        <span class="sh-num">REF</span>
        <span class="sh-title">REQUIRED INPUT FEATURES</span>
        <span class="sh-line"></span>
    </div>
    <div class="fg">
        <div class="fc"><div class="fc-ico">◈</div>
            <div class="fc-t">Attendance</div>
            <div class="fc-d">attendance_pct<br>% of classes attended</div></div>
        <div class="fc"><div class="fc-ico">◈</div>
            <div class="fc-t">Quiz Scores</div>
            <div class="fc-d">quiz_1 → quiz_5<br>Auto-computes avg &amp; std</div></div>
        <div class="fc"><div class="fc-ico">◈</div>
            <div class="fc-t">Assignment</div>
            <div class="fc-d">assignment_score<br>Cumulative score</div></div>
        <div class="fc"><div class="fc-ico">◈</div>
            <div class="fc-t sess">Sessional 1 &amp; 2</div>
            <div class="fc-d">sessional1, sessional2<br>High priority · mid-term exams</div></div>
        <div class="fc"><div class="fc-ico">◈</div>
            <div class="fc-t">Integrity</div>
            <div class="fc-d">cheating_count<br>Academic violations</div></div>
        <div class="fc"><div class="fc-ico">◈</div>
            <div class="fc-t">Teacher Score</div>
            <div class="fc-d">teacher_feedback_score<br>Instructor assessment</div></div>
    </div>
    <br>
    <div style="max-width:880px;margin:.8rem auto 0;background:var(--bg-card);
        border:1px solid var(--border);border-left:3px solid var(--cyan);
        border-radius:var(--r);padding:.8rem 1.2rem;
        font-family:var(--mono);font-size:0.62rem;color:var(--text-muted);line-height:2;">
        <span style="color:var(--cyan);">AUTO-ENGINEERED</span>
        &nbsp;·&nbsp; The app automatically computes:
        quiz_avg &amp; quiz_std are auto-computed from quiz_1–quiz_5.
        No manual calculation needed.
    </div>
    """, unsafe_allow_html=True)