"""
Crop Recommender — AgroSage
Predict the best crop for your soil and weather conditions.
"""

import os

import streamlit as st
import plotly.graph_objects as go

from utils import load_css, render_sidebar
from utils.crop_predictor import load_crop_models, predict_crop


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
_MODEL_FILES = [
    "crop_model.pkl",
    "crop_scaler.pkl",
    "crop_classes.pkl",
    "crop_kmeans.pkl",
    "crop_cluster_map.pkl",
]


def _crop_model_signature() -> tuple[float, ...]:
    sig = []
    for fname in _MODEL_FILES:
        path = os.path.join(MODEL_DIR, fname)
        sig.append(os.path.getmtime(path) if os.path.isfile(path) else 0.0)
    return tuple(sig)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Crop Recommender — AgroSage",
    page_icon="Ag",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
render_sidebar()

# ── Page-specific styles ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Section label ── */
.cr-section {
    margin-top: 1.75rem;
    margin-bottom: 0.5rem;
}
.cr-section-title {
    font-family: var(--font-body);
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--color-border-light);
}

/* ── Best crop result banner ── */
.cr-best {
    background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.75rem;
    color: #FFFFFF;
    box-shadow: var(--shadow-lg);
}
.cr-best-eyebrow {
    font-family: var(--font-body);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.7;
    margin-bottom: 0.3rem;
}
.cr-best-crop {
    font-family: var(--font-body);
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    text-transform: capitalize;
    color: #FFFFFF !important;
}
.cr-best-confidence {
    font-family: var(--font-mono);
    font-size: 0.88rem;
    opacity: 0.8;
    margin-top: 0.3rem;
}

/* ── Compact stat card ── */
.cr-stat {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    transition: var(--transition);
}
.cr-stat:hover {
    border-color: var(--color-accent);
    box-shadow: var(--shadow-sm);
}
.cr-stat-label {
    font-family: var(--font-body);
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.15rem;
}
.cr-stat-value {
    font-family: var(--font-mono);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--color-primary);
    text-transform: capitalize;
}

/* ── Alternative pill ── */
.cr-alt {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    padding: 0.65rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.35rem;
    transition: var(--transition);
}
.cr-alt:hover {
    border-color: var(--color-accent);
}
.cr-alt-name {
    font-family: var(--font-body);
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--color-text);
    text-transform: capitalize;
}
.cr-alt-prob {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--color-accent);
}

/* ── Similar crop tag ── */
.cr-tag {
    display: inline-block;
    font-family: var(--font-body);
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.25rem 0.7rem;
    border-radius: 100px;
    background: var(--color-accent-glow);
    color: var(--color-primary-light);
    margin: 0.2rem 0.25rem 0.2rem 0;
    text-transform: capitalize;
}

/* ── Note card ── */
.cr-note {
    background: var(--color-surface-alt);
    border: 1px solid var(--color-border-light);
    border-left: 3px solid var(--color-accent);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    margin-top: 0.75rem;
}
.cr-note-title {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}
.cr-note-text {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--color-text-secondary);
    line-height: 1.5;
    margin: 0;
}

/* ── Input form layout ── */
.cr-form-group {
    margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="margin-bottom:0.25rem;">
        <span class="ag-badge ag-badge-success" style="margin-bottom:0.5rem;">ML Model</span>
        <h1 style="font-size:1.5rem !important; font-weight:700 !important;
                   margin:0.35rem 0 0 0 !important; padding:0 !important;">
            Crop Recommender
        </h1>
        <p style="font-size:0.84rem; color:var(--color-text-secondary);
                  margin:0.15rem 0 0 0; font-weight:400;">
            Enter soil nutrient levels and climate data to get a data-driven crop recommendation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_models(_signature: tuple[float, ...]):
    """Cache model artifacts across reruns."""
    return load_crop_models()


try:
    _load_models(_crop_model_signature())
except FileNotFoundError as exc:
    st.error(
        "**Models not found.** Train the crop model before using this page.\n\n"
        "```bash\npython training/train_crop_model.py\n```\n\n"
        f"Details: {exc}"
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(
        "An unexpected error occurred while loading crop models. "
        "Please verify model artifacts and retrain if needed.\n\n"
        f"Details: {type(exc).__name__}: {exc}"
    )
    st.stop()


if "crop_result" not in st.session_state:
    st.session_state["crop_result"] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="cr-section"><div class="cr-section-title">Input Parameters</div></div>',
    unsafe_allow_html=True,
)

with st.form("crop_form", clear_on_submit=False):

    # ── Row 1: Soil nutrients ──
    st.markdown(
        "<p style='font-size:0.78rem; font-weight:600; color:var(--color-primary); "
        "margin:0 0 0.35rem 0;'>Soil Nutrients</p>",
        unsafe_allow_html=True,
    )
    n_col, p_col, k_col = st.columns(3)
    with n_col:
        nitrogen = st.number_input(
            "Nitrogen (N)", min_value=0, max_value=140,
            value=50, step=1, help="Nitrogen content in soil (mg/kg)",
        )
    with p_col:
        phosphorus = st.number_input(
            "Phosphorus (P)", min_value=5, max_value=145,
            value=53, step=1, help="Phosphorus content in soil (mg/kg)",
        )
    with k_col:
        potassium = st.number_input(
            "Potassium (K)", min_value=5, max_value=205,
            value=48, step=1, help="Potassium content in soil (mg/kg)",
        )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # ── Row 2: Climate conditions ──
    st.markdown(
        "<p style='font-size:0.78rem; font-weight:600; color:var(--color-primary); "
        "margin:0 0 0.35rem 0;'>Climate Conditions</p>",
        unsafe_allow_html=True,
    )
    t_col, h_col = st.columns(2)
    with t_col:
        temperature = st.slider(
            "Temperature (C)", min_value=8.0, max_value=45.0,
            value=25.0, step=0.5, help="Average temperature in degrees Celsius",
        )
    with h_col:
        humidity = st.slider(
            "Humidity (%)", min_value=10.0, max_value=100.0,
            value=70.0, step=0.5, help="Average relative humidity percentage",
        )

    ph_col, rain_col = st.columns(2)
    with ph_col:
        ph = st.slider(
            "Soil pH", min_value=3.5, max_value=10.0,
            value=6.5, step=0.1, format="%.1f",
            help="pH level of the soil (3.5 = acidic, 10 = alkaline)",
        )
    with rain_col:
        rainfall = st.number_input(
            "Rainfall (mm)", min_value=20.0, max_value=300.0,
            value=100.0, step=5.0, help="Annual rainfall in millimeters",
        )

    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "Get Recommendation", use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

if submitted:
    with st.spinner("Analysing soil and climate data..."):
        result = predict_crop(
            nitrogen=float(nitrogen),
            phosphorus=float(phosphorus),
            potassium=float(potassium),
            temperature=float(temperature),
            humidity=float(humidity),
            ph=float(ph),
            rainfall=float(rainfall),
        )
    st.session_state["crop_result"] = result


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

result = st.session_state.get("crop_result")

if result is None:
    # No prediction yet — show helper
    st.markdown(
        """
        <div class="cr-note" style="margin-top:1.5rem;">
            <div class="cr-note-title">How to use</div>
            <p class="cr-note-text">
                Adjust the soil nutrient levels and climate parameters above, then
                press <strong>Get Recommendation</strong> to receive a crop suggestion
                based on the trained Random Forest model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if "error" in result:
    st.error(result["error"])
    st.stop()


# ── Section: Results ──
st.markdown(
    '<div class="cr-section"><div class="cr-section-title">Recommendation</div></div>',
    unsafe_allow_html=True,
)

best_crop = str(result.get("best_crop", "unknown"))
confidence = float(result.get("confidence", 0.0))
alternatives = result.get("alternatives", [])
similar_crops = result.get("similar_crops", [])
all_probs = result.get("all_probabilities", {})

# ── Debug panel: Top probabilities ──
top_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)[:5]
with st.expander("Prediction Debug (Top 5 Probabilities)", expanded=False):
    st.caption("Use this to verify model confidence distribution for the current input.")
    for rank, (crop_name, prob) in enumerate(top_probs, start=1):
        st.markdown(f"**{rank}. {crop_name}** - {prob:.2f}%")
        st.progress(max(0.0, min(1.0, prob / 100.0)))


# ── 1. Best crop banner ──
st.markdown(
    f"""
    <div class="cr-best">
        <div class="cr-best-eyebrow">Recommended Crop</div>
        <div class="cr-best-crop">{best_crop}</div>
        <div class="cr-best-confidence">Confidence: {confidence:.1f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)


# ── 2. Stat cards row ──
sc1, sc2, sc3, sc4 = st.columns(4)

with sc1:
    st.markdown(
        f"""
        <div class="cr-stat">
            <div class="cr-stat-label">Best Match</div>
            <div class="cr-stat-value">{best_crop}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sc2:
    st.markdown(
        f"""
        <div class="cr-stat">
            <div class="cr-stat-label">Confidence</div>
            <div class="cr-stat-value">{confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sc3:
    runner_up = alternatives[0]["name"] if alternatives else "—"
    st.markdown(
        f"""
        <div class="cr-stat">
            <div class="cr-stat-label">Runner Up</div>
            <div class="cr-stat-value">{runner_up}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sc4:
    cluster_count = len(similar_crops)
    st.markdown(
        f"""
        <div class="cr-stat">
            <div class="cr-stat-label">Similar Crops</div>
            <div class="cr-stat-value">{cluster_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)


# ── 3. Alternatives & similar crops (two columns) ──
alt_col, sim_col = st.columns(2)

with alt_col:
    st.markdown(
        '<div class="cr-section"><div class="cr-section-title">Top Alternatives</div></div>',
        unsafe_allow_html=True,
    )
    if alternatives:
        for alt in alternatives:
            prob_display = f"{alt['probability']:.1f}%" if alt["probability"] >= 0.1 else "<0.1%"
            st.markdown(
                f"""
                <div class="cr-alt">
                    <span class="cr-alt-name">{alt['name']}</span>
                    <span class="cr-alt-prob">{prob_display}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<p style='font-size:0.84rem; color:var(--color-text-muted);'>No alternatives available.</p>",
            unsafe_allow_html=True,
        )

with sim_col:
    st.markdown(
        '<div class="cr-section"><div class="cr-section-title">Similar Crops (Same Cluster)</div></div>',
        unsafe_allow_html=True,
    )
    if similar_crops:
        tags_html = "".join(f'<span class="cr-tag">{c}</span>' for c in similar_crops)
        st.markdown(
            f'<div style="padding-top:0.2rem;">{tags_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='font-size:0.84rem; color:var(--color-text-muted);'>No similar crops in this cluster.</p>",
            unsafe_allow_html=True,
        )


# ── 4. Probability chart ──
st.markdown(
    '<div class="cr-section"><div class="cr-section-title">Full Probability Distribution</div></div>',
    unsafe_allow_html=True,
)

# Sort by probability descending, show only crops with > 0%
sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
# Show top 10 or all with > 0
display_probs = [(c, p) for c, p in sorted_probs if p > 0]
if len(display_probs) < 3:
    display_probs = sorted_probs[:5]  # show at least top 5

crop_names = [c.capitalize() for c, _ in reversed(display_probs)]
crop_vals = [p for _, p in reversed(display_probs)]

# Color bars: highlight the best crop
bar_colors = [
    "#40916C" if c.lower() == best_crop.lower() else "#D5DDD9"
    for c, _ in reversed(display_probs)
]

fig = go.Figure(
    go.Bar(
        x=crop_vals,
        y=crop_names,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(width=0),
            cornerradius=4,
        ),
        text=[f"{v:.1f}%" for v in crop_vals],
        textposition="outside",
        textfont=dict(
            family="DM Mono, monospace",
            size=11,
            color="#5A6B60",
        ),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    )
)

fig.update_layout(
    height=max(240, len(display_probs) * 32 + 60),
    margin=dict(l=10, r=40, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=12, color="#1A1F1C"),
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, max(crop_vals) * 1.25] if crop_vals else [0, 100],
    ),
    yaxis=dict(
        showgrid=False,
        tickfont=dict(family="DM Sans, sans-serif", size=12, color="#5A6B60"),
        automargin=True,
    ),
    bargap=0.25,
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── 5. Agronomic note ──
st.markdown(
    f"""
    <div class="cr-note">
        <div class="cr-note-title">Agronomic Note</div>
        <p class="cr-note-text">
            This recommendation is based on a Random Forest classifier trained on
            2,200 soil and climate samples across 22 crop types. The model achieved
            99.55% accuracy on the test set. For best results, combine this suggestion
            with local agronomic expertise, soil testing reports, and seasonal planting
            calendars for your region. The <strong>{best_crop}</strong> prediction
            is based on the specific N-P-K, temperature, humidity, pH, and rainfall
            values you provided.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
