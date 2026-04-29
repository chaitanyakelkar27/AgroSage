"""
AgroSage — Home Dashboard
===========================
Main entry-point for the Streamlit multi-page app.
Run with:  streamlit run Home.py
"""

import os
import datetime

import streamlit as st
import pandas as pd

from utils import load_css, render_sidebar

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the very first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgroSage — Smart Crop Advisory",
    page_icon="Ag",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
render_sidebar()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD-SPECIFIC CSS  (supplements global style.css)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Section spacing ── */
.ag-section {
    margin-top: 2rem;
    margin-bottom: 0.5rem;
}

.ag-section-title {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--color-border-light);
}

/* ── Hero banner ── */
.ag-hero {
    background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-light) 60%, var(--color-accent) 100%);
    border-radius: var(--radius-lg);
    padding: 2rem 2.25rem;
    color: #FFFFFF;
    box-shadow: var(--shadow-lg);
    margin-bottom: 1.75rem;
}

.ag-hero-eyebrow {
    font-family: var(--font-body);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.7;
    margin-bottom: 0.35rem;
}

.ag-hero-title {
    font-family: var(--font-body);
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.25;
    margin-bottom: 0.5rem;
    color: #FFFFFF !important;
}

.ag-hero-desc {
    font-family: var(--font-body);
    font-size: 0.88rem;
    line-height: 1.55;
    opacity: 0.82;
    max-width: 600px;
}

/* ── Feature module cards ── */
.ag-module-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-lg);
    padding: 1.35rem 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
    height: 100%;
}

.ag-module-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--color-accent);
    transform: translateY(-2px);
}

.ag-module-icon {
    width: 36px;
    height: 36px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.92rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.85rem;
    flex-shrink: 0;
}

.ag-module-icon.crop    { background: linear-gradient(135deg, #2D6A4F, #40916C); }
.ag-module-icon.disease { background: linear-gradient(135deg, #1B6B93, #4FC0D0); }
.ag-module-icon.weather { background: linear-gradient(135deg, #6C3483, #A569BD); }
.ag-module-icon.assistant { background: linear-gradient(135deg, #6C5B2A, #C8A24A); }

.ag-module-name {
    font-family: var(--font-body);
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--color-primary);
    margin-bottom: 0.3rem;
}

.ag-module-desc {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--color-text-secondary);
    line-height: 1.5;
    margin: 0;
}

/* ── Status row ── */
.ag-status-row {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: var(--transition);
}

.ag-status-row:hover {
    border-color: var(--color-border);
}

.ag-status-label {
    font-family: var(--font-body);
    font-size: 0.84rem;
    font-weight: 500;
    color: var(--color-text);
}

.ag-status-meta {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.ag-status-detail {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--color-text-muted);
}

/* ── Footer ── */
.ag-footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid var(--color-border-light);
    margin-top: 2.5rem;
}

.ag-footer p {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--color-text-muted);
    margin: 0;
    letter-spacing: 0.02em;
}

/* ── Override Streamlit default sidebar page links text ── */
[data-testid="stSidebar"] .stPageLink p {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Guide step cards ── */
.ag-guide-step {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: var(--transition);
}

.ag-guide-step:hover {
    border-color: var(--color-accent);
}

.ag-guide-step-num {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--color-accent);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.2rem;
}

.ag-guide-step-title {
    font-family: var(--font-body);
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--color-primary);
    margin-bottom: 0.25rem;
}

.ag-guide-step-desc {
    font-family: var(--font-body);
    font-size: 0.8rem;
    color: var(--color-text-secondary);
    line-height: 1.5;
    margin: 0;
}

.ag-guide-step-cmd {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    background: var(--color-surface-alt);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.65rem;
    margin-top: 0.45rem;
    display: inline-block;
    color: var(--color-primary);
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _file_exists(path: str) -> bool:
    return os.path.isfile(path)


def _file_size_str(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _modified_ago(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    delta = datetime.datetime.now() - mtime
    if delta.days > 0:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    minutes = delta.seconds // 60
    return f"{minutes}m ago" if minutes > 0 else "just now"


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="margin-bottom:0.4rem;">
        <h1 style="font-size:1.5rem !important; font-weight:700 !important;
                   margin:0 !important; padding:0 !important;">
            Dashboard
        </h1>
        <p style="font-size:0.84rem; color:var(--color-text-secondary);
                  margin:0.15rem 0 0 0; font-weight:400;">
            System overview and quick access to all modules
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="ag-hero">
        <div class="ag-hero-eyebrow">Welcome to AgroSage</div>
        <div class="ag-hero-title">Smart Farming Starts Here</div>
        <div class="ag-hero-desc">
            Combine soil nutrient analysis, real-time weather data, and
            deep-learning disease detection to make data-driven crop decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM METRICS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="ag-section"><div class="ag-section-title">Overview</div></div>',
            unsafe_allow_html=True)

crop_csv = os.path.join(DATA_DIR, "Crop_recommendation.csv")
if _file_exists(crop_csv):
    try:
        _df = pd.read_csv(crop_csv)
        n_samples = len(_df)
        n_crops = _df["label"].str.strip().nunique()
        n_features = 7
    except Exception:
        n_samples, n_crops, n_features = 0, 0, 0
else:
    n_samples, n_crops, n_features = 0, 0, 0

model_files = ["crop_model.pkl", "crop_scaler.pkl", "crop_classes.pkl",
               "crop_kmeans.pkl", "crop_cluster_map.pkl"]
models_ready = sum(1 for f in model_files if _file_exists(os.path.join(MODEL_DIR, f)))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training Samples", f"{n_samples:,}" if n_samples else "—")
col2.metric("Crop Classes", str(n_crops) if n_crops else "—")
col3.metric("Input Features", str(n_features) if n_features else "—")
col4.metric("Model Artifacts", f"{models_ready}/{len(model_files)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLATFORM MODULES
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="ag-section"><div class="ag-section-title">Platform Modules</div></div>',
            unsafe_allow_html=True)

mc1, mc2, mc3, mc4 = st.columns(4)

with mc1:
    st.markdown(
        """
        <div class="ag-module-card">
            <div class="ag-module-icon crop">CR</div>
            <div class="ag-module-name">Crop Recommender</div>
            <p class="ag-module-desc">
                Enter soil and weather parameters to receive crop
                recommendations powered by a Random Forest ensemble model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mc2:
    st.markdown(
        """
        <div class="ag-module-card">
            <div class="ag-module-icon disease">DD</div>
            <div class="ag-module-name">Disease Detector</div>
            <p class="ag-module-desc">
                Upload a leaf photograph for CNN-based disease classification
                with confidence scores and suggested treatments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mc3:
    st.markdown(
        """
        <div class="ag-module-card">
            <div class="ag-module-icon weather">WA</div>
            <div class="ag-module-name">Weather Analyst</div>
            <p class="ag-module-desc">
                Access real-time weather data and explore historical climate
                trends through interactive charts for your region.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mc4:
    st.markdown(
        """
        <div class="ag-module-card">
            <div class="ag-module-icon assistant">QA</div>
            <div class="ag-module-name">Agronomy Assistant</div>
            <p class="ag-module-desc">
                Ask agronomy questions or get help understanding AgroSage workflows
                with the LangChain Q&A assistant.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM STATUS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="ag-section"><div class="ag-section-title">System Status</div></div>',
            unsafe_allow_html=True)

status_items = [
    ("Crop Dataset",              crop_csv),
    ("Climate Dataset",           os.path.join(DATA_DIR, "DailyDelhiClimate.csv")),
    ("Random Forest Model",       os.path.join(MODEL_DIR, "crop_model.pkl")),
    ("Feature Scaler",            os.path.join(MODEL_DIR, "crop_scaler.pkl")),
    ("Crop Classes",              os.path.join(MODEL_DIR, "crop_classes.pkl")),
    ("KMeans Clustering Model",   os.path.join(MODEL_DIR, "crop_kmeans.pkl")),
    ("Cluster Map",               os.path.join(MODEL_DIR, "crop_cluster_map.pkl")),
]

s1, s2 = st.columns(2)

for idx, (label, filepath) in enumerate(status_items):
    col = s1 if idx % 2 == 0 else s2
    exists = _file_exists(filepath)
    badge_cls = "ag-badge-success" if exists else "ag-badge-danger"
    badge_txt = "Ready" if exists else "Missing"
    meta = ""
    if exists:
        size = _file_size_str(filepath)
        age = _modified_ago(filepath)
        meta = f'{size} &middot; {age}'

    with col:
        st.markdown(
            f"""
            <div class="ag-status-row">
                <span class="ag-status-label">{label}</span>
                <span class="ag-status-meta">
                    <span class="ag-status-detail">{meta}</span>
                    <span class="ag-badge {badge_cls}">{badge_txt}</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Summary
missing = [lbl for lbl, fp in status_items if not _file_exists(fp)]
if not missing:
    st.success("All systems operational — datasets loaded, models trained and ready.")
else:
    st.warning(
        f"**{len(missing)} component(s) missing:** {', '.join(missing)}. "
        "Follow the setup guide below."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  GETTING STARTED
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="ag-section"><div class="ag-section-title">Getting Started</div></div>',
            unsafe_allow_html=True)

guide_steps = [
    {
        "num": "Step 01",
        "title": "Install dependencies",
        "desc": "Install all required Python packages including Streamlit, "
                "scikit-learn, TensorFlow, and Plotly.",
        "cmd": "pip install -r requirements.txt",
    },
    {
        "num": "Step 02",
        "title": "Place datasets",
        "desc": "Add Crop_recommendation.csv (2,200 rows, 7 features, 22 crop classes) "
                "and DailyDelhiClimate.csv into the data/ folder.",
        "cmd": None,
    },
    {
        "num": "Step 03",
        "title": "Train the crop model",
        "desc": "Runs GridSearchCV on a Random Forest classifier, fits a scaler and "
                "KMeans model, then saves all artifacts to models/.",
        "cmd": "python training/train_crop_model.py",
    },
    {
        "num": "Step 04",
        "title": "Configure environment",
        "desc": "Copy .env.example to .env and add your OpenWeatherMap API key. "
                "Only required for the Weather Analyst page.",
        "cmd": "cp .env.example .env",
    },
    {
        "num": "Step 05",
        "title": "Launch the application",
        "desc": "Starts the Streamlit server and opens AgroSage in your browser.",
        "cmd": "streamlit run Home.py",
    },
]

g1, g2 = st.columns(2)
for idx, step in enumerate(guide_steps):
    col = g1 if idx % 2 == 0 else g2
    cmd_html = (
        f'<div class="ag-guide-step-cmd">{step["cmd"]}</div>'
        if step["cmd"] else ""
    )
    with col:
        st.markdown(
            f"""
            <div class="ag-guide-step">
                <div class="ag-guide-step-num">{step["num"]}</div>
                <div class="ag-guide-step-title">{step["title"]}</div>
                <p class="ag-guide-step-desc">{step["desc"]}</p>
                {cmd_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="ag-footer">
        <p>AgroSage v1.0.0 &middot; Streamlit &middot; scikit-learn &middot; TensorFlow &middot; Plotly</p>
    </div>
    """,
    unsafe_allow_html=True,
)
