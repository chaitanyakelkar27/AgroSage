"""
Disease Detector — AgroSage
Upload a leaf image to detect plant diseases with AI-powered analysis.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from utils import load_css, render_sidebar
from utils.disease_predictor import (
    load_disease_models,
    load_vit_model,
    preprocess_image,
    clean_image,
    prepare_classifier_input,
    compute_image_quality_metrics,
    parse_disease_label,
    predict_disease,
    predict_disease_vit,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Disease Detector — AgroSage",
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
.dd-section {
    margin-top: 1.75rem;
    margin-bottom: 0.5rem;
}
.dd-section-title {
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

/* ── Upload panel ── */
.dd-upload-panel {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow-sm);
}

/* ── Alert cards ── */
.dd-alert {
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.75rem;
    color: #FFFFFF;
    box-shadow: var(--shadow-lg);
}
.dd-alert.high {
    background: linear-gradient(135deg, #C0392B 0%, #E74C3C 100%);
}
.dd-alert.medium {
    background: linear-gradient(135deg, #D35400 0%, #E67E22 100%);
}
.dd-alert.low {
    background: linear-gradient(135deg, #2980B9 0%, #3498DB 100%);
}
.dd-alert.none {
    background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%);
}
.dd-alert-eyebrow {
    font-family: var(--font-body);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.75;
    margin-bottom: 0.3rem;
}
.dd-alert-title {
    font-family: var(--font-body);
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
    margin-bottom: 0.15rem;
}
.dd-alert-sub {
    font-family: var(--font-mono);
    font-size: 0.88rem;
    opacity: 0.8;
    margin-top: 0.3rem;
}

/* ── Compact stat card ── */
.dd-stat {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    transition: var(--transition);
}
.dd-stat:hover {
    border-color: var(--color-accent);
    box-shadow: var(--shadow-sm);
}
.dd-stat-label {
    font-family: var(--font-body);
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.15rem;
}
.dd-stat-value {
    font-family: var(--font-mono);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--color-primary);
}

/* ── Top predictions row ── */
.dd-pred-row {
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
.dd-pred-row:hover {
    border-color: var(--color-accent);
}
.dd-pred-rank {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--color-text-muted);
    font-weight: 500;
    min-width: 28px;
}
.dd-pred-label {
    font-family: var(--font-body);
    font-size: 0.86rem;
    font-weight: 500;
    color: var(--color-text);
    flex: 1;
    margin-left: 0.5rem;
}
.dd-pred-prob {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--color-accent);
}

/* ── Note card ── */
.dd-note {
    background: var(--color-surface-alt);
    border: 1px solid var(--color-border-light);
    border-left: 3px solid var(--color-accent);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    margin-top: 0.75rem;
}
.dd-note-title {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}
.dd-note-text {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--color-text-secondary);
    line-height: 1.5;
    margin: 0;
}

/* ── Risk level badge ── */
.dd-risk-badge {
    display: inline-block;
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.65rem;
    border-radius: 100px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.dd-risk-badge.high   { background: rgba(192, 57, 43, 0.12); color: #C0392B; }
.dd-risk-badge.medium { background: rgba(230, 126, 34, 0.12); color: #E67E22; }
.dd-risk-badge.low    { background: rgba(41, 128, 185, 0.12); color: #2980B9; }
.dd-risk-badge.none   { background: rgba(39, 174, 96, 0.12); color: #27AE60; }

/* ── Cleaned image preview ── */
.dd-preview-container {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.dd-preview-box {
    flex: 1;
    text-align: center;
}
.dd-preview-label {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="margin-bottom:0.25rem;">
        <span class="ag-badge ag-badge-info" style="margin-bottom:0.5rem;">Deep Learning</span>
        <h1 style="font-size:1.5rem !important; font-weight:700 !important;
                   margin:0.35rem 0 0 0 !important; padding:0 !important;">
            Disease Detector
        </h1>
        <p style="font-size:0.84rem; color:var(--color-text-secondary);
                  margin:0.15rem 0 0 0; font-weight:400;">
            Upload a leaf photograph for CNN-based disease classification with confidence analysis
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
def _load_models():
    """Cache disease model artifacts across reruns."""
    bundle = {
        "cnn_bundle": load_disease_models(),
        "vit_bundle": None,
        "vit_error": None,
    }

    try:
        bundle["vit_bundle"] = load_vit_model()
    except Exception as exc:  # noqa: BLE001
        bundle["vit_error"] = exc

    return bundle


try:
    models = _load_models()
except FileNotFoundError as exc:
    st.error(
        "**Disease models not found.** Train the disease model before using this page.\n\n"
        "```bash\npython training/train_disease_model.py\n```\n\n"
        f"Details: {exc}"
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(
        "An unexpected error occurred while loading disease models. "
        "Please verify model artifacts and retrain if needed.\n\n"
        f"Details: {type(exc).__name__}: {exc}"
    )
    st.stop()

cnn_bundle = models["cnn_bundle"]
autoencoder = cnn_bundle["autoencoder"]
cnn = cnn_bundle["cnn"]
idx_to_class = cnn_bundle["idx_to_class"]
vit_bundle = models.get("vit_bundle")
vit_error = models.get("vit_error")

# ── Session state initialisation ──────────────────────────────────────────────
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None
if "disease_result" not in st.session_state:
    st.session_state["disease_result"] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="dd-section"><div class="dd-section-title">Image Upload</div></div>',
    unsafe_allow_html=True,
)

upload_col, preview_col = st.columns([3, 2])

with upload_col:
    st.markdown('<div class="dd-upload-panel">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WebP. For best results use a clear, close-up photograph of the leaf.",
        key="disease_uploader",
    )
    model_choice = st.radio(
        "Model",
        options=["CNN (default)", "ViT (optional)"],
        horizontal=True,
        help=(
            "CNN uses the existing autoencoder + classifier. "
            "ViT requires a fine-tuned model in models/vit and extra dependencies."
        ),
    )
    field_photo_mode = st.toggle(
        "Field Photo Mode (recommended)",
        value=True,
        help=(
            "Uses test-time augmentation and stricter false-positive checks for real-world "
            "photos that include natural background, shadows, or hands."
        ),
    )
    st.markdown('</div>', unsafe_allow_html=True)

with preview_col:
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.session_state["uploaded_image"] = pil_image
        st.image(pil_image, caption="Uploaded Leaf Image", use_container_width=True)
    elif st.session_state["uploaded_image"] is not None:
        st.image(st.session_state["uploaded_image"], caption="Uploaded Leaf Image", use_container_width=True)
    else:
        st.markdown(
            """
            <div class="dd-note" style="margin-top:0;">
                <div class="dd-note-title">Preview</div>
                <p class="dd-note-text">
                    Your uploaded leaf image will appear here.
                    Use a clear photo of a single leaf for best results.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS TRIGGER
# ═══════════════════════════════════════════════════════════════════════════════

current_image = st.session_state.get("uploaded_image")

if current_image is not None:
    analyse_btn = st.button("Analyse Leaf", use_container_width=True, key="analyse_btn")

    if analyse_btn:
        with st.spinner("Processing image through inference pipeline..."):
            # Determine input shape from CNN
            cnn_input_shape = cnn.input_shape
            if cnn_input_shape and len(cnn_input_shape) >= 3:
                target_h = cnn_input_shape[1] or 128
                target_w = cnn_input_shape[2] or 128
            else:
                target_h, target_w = 128, 128

            target_size = (target_w, target_h)

            # Preprocess
            img_batch = preprocess_image(current_image, target_size)

            # Use model-compatible inference input to avoid denoising twice.
            classifier_input, cleaned_batch = prepare_classifier_input(
                cnn, autoencoder, img_batch
            )

            reconstruction_error = float(np.mean((img_batch - cleaned_batch) ** 2))
            quality_metrics = compute_image_quality_metrics(img_batch)

            if model_choice.startswith("ViT"):
                if vit_bundle is None:
                    st.error(
                        "**ViT model unavailable.** Place a fine-tuned transformers ViT model in "
                        "`models/vit` and install torch, torchvision, and transformers.\n\n"
                        f"Details: {vit_error}"
                    )
                    st.stop()

                result = predict_disease_vit(
                    vit_bundle["vit"],
                    vit_bundle["vit_processor"],
                    vit_bundle["vit_id_to_label"],
                    current_image,
                    idx_to_class,
                    reconstruction_error=reconstruction_error,
                    image_quality_metrics=quality_metrics,
                )
            else:
                # Predict with existing CNN
                result = predict_disease(
                    cnn,
                    classifier_input,
                    idx_to_class,
                    reconstruction_error=reconstruction_error,
                    image_quality_metrics=quality_metrics,
                    inference_mode="field" if field_photo_mode else "standard",
                )

            # Store cleaned image for preview
            cleaned_arr = (np.squeeze(cleaned_batch, axis=0) * 255).astype(np.uint8)
            result["cleaned_image"] = cleaned_arr

            # Store original array for preview
            original_arr = (np.squeeze(img_batch, axis=0) * 255).astype(np.uint8)
            result["original_preview"] = original_arr

            st.session_state["disease_result"] = result

else:
    # No image uploaded — show helper
    st.markdown(
        """
        <div class="dd-note" style="margin-top:1.5rem;">
            <div class="dd-note-title">How to use</div>
            <p class="dd-note-text">
                Upload a clear photograph of a plant leaf using the file uploader above,
                then press <strong>Analyse Leaf</strong> to run the image through the
                autoencoder denoiser and CNN classifier.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

result = st.session_state.get("disease_result")

if result is None:
    st.info("Press **Analyse Leaf** above to classify the uploaded image.")
    st.stop()


plant = result["plant"]
disease = result["disease"]
probability = result["probability"]
alert_level = result["alert_level"]
all_results = result["all_results"]
requires_manual_review = bool(result.get("requires_manual_review", False))
likely_false_positive = bool(result.get("likely_false_positive", False))
is_inconclusive = bool(result.get("is_inconclusive", False))
diagnostic_flags = result.get("diagnostic_flags", [])
healthy_probability_for_plant = float(result.get("healthy_probability_for_plant", 0.0))
confidence_gap = float(result.get("confidence_gap", 0.0))
top_consistency = float(result.get("top_consistency", 1.0))
top_probability_std = float(result.get("top_probability_std", 0.0))
inference_mode = str(result.get("inference_mode", "field"))
reconstruction_error = result.get("reconstruction_error")
image_quality_metrics = result.get("image_quality_metrics", {})

# Determine alert text
_alert_labels = {
    "high": "High Risk — Immediate Attention Required",
    "medium": "Medium Risk — Monitor Closely",
    "low": "Low Risk — Likely Minor Issue",
    "none": "Healthy — No Disease Detected",
}
alert_text = _alert_labels.get(alert_level, "Analysis Complete")
if is_inconclusive and alert_level != "none":
    alert_text = "Inconclusive — Retake Photo Or Verify Manually"
elif likely_false_positive and alert_level != "none":
    alert_text = "Possible False Positive — Visual Symptoms Are Weak"
elif requires_manual_review and alert_level != "none":
    alert_text = "Confidence Reduced — Manual Review Recommended"
_risk_badge_label = {
    "high": "High Risk",
    "medium": "Medium Risk",
    "low": "Low Risk",
    "none": "Healthy",
}


# ── 1. Risk Alert Banner ──

st.markdown(
    '<div class="dd-section"><div class="dd-section-title">Diagnosis</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="dd-alert {alert_level}">
        <div class="dd-alert-eyebrow">{alert_text}</div>
        <div class="dd-alert-title">{plant} — {disease}</div>
        <div class="dd-alert-sub">Confidence: {probability:.1%}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

if requires_manual_review:
    review_points = []
    if "high_reconstruction_error" in diagnostic_flags:
        review_points.append("Image characteristics differ from the model's training-domain pattern.")
    if "high_background_complexity" in diagnostic_flags:
        review_points.append("Background detail is high, which can reduce model reliability on field photos.")
    if "very_low_leaf_coverage" in diagnostic_flags:
        review_points.append("Leaf occupies a small part of the frame, so diagnosis confidence is reduced.")
    if "low_leaf_coverage" in diagnostic_flags:
        review_points.append("Leaf coverage is limited in this frame, which may affect confidence.")
    if "weak_lesion_evidence" in diagnostic_flags:
        review_points.append(
            "The image appears mostly green with weak visible lesion evidence for the predicted disease class."
        )
    if "healthy_competition" in diagnostic_flags:
        review_points.append(
            "Healthy class probability is competitive with disease probability for this plant."
        )
    if "unstable_tta_prediction" in diagnostic_flags:
        review_points.append(
            "Prediction changed across augmentations, indicating low inference stability in field mode."
        )
    if "high_tta_variance" in diagnostic_flags:
        review_points.append(
            "Top class confidence varied across augmentations, indicating uncertain diagnosis."
        )
    if not review_points:
        review_points.append("Model reliability checks suggest this prediction may require manual confirmation.")

    metrics_line_parts = []
    if reconstruction_error is not None:
        metrics_line_parts.append(f"Reconstruction error: {float(reconstruction_error):.6f}")
    edge_density = image_quality_metrics.get("edge_density")
    if edge_density is not None:
        metrics_line_parts.append(f"Edge density: {float(edge_density):.4f}")
    green_ratio = image_quality_metrics.get("green_ratio")
    if green_ratio is not None:
        metrics_line_parts.append(f"Leaf coverage: {float(green_ratio):.1%}")
    lesion_score = image_quality_metrics.get("lesion_evidence_score")
    if lesion_score is not None:
        metrics_line_parts.append(f"Lesion evidence score: {float(lesion_score):.4f}")
    metrics_line_parts.append(f"Inference mode: {inference_mode}")
    metrics_line_parts.append(f"TTA top consistency: {top_consistency:.1%}")
    metrics_line_parts.append(f"TTA top std-dev: {top_probability_std:.4f}")
    metrics_line = " | ".join(metrics_line_parts)

    banner_line = "Result marked as inconclusive." if is_inconclusive else "Prediction quality guardrail triggered."
    st.warning(
        "\n".join([
            banner_line,
            "Hands or some background are normal in real-world photos; this is only a confidence warning.",
            *[f"- {point}" for point in review_points],
            f"- Healthy probability for {plant}: {healthy_probability_for_plant:.1%}",
            f"- Confidence gap (top1-top2): {confidence_gap:.1%}",
            *( [f"- {metrics_line}"] if metrics_line else [] ),
            "Optional improvement: keep one leaf as the main focus under natural daylight.",
        ])
    )


# ── 2. Stat cards row ──

sc1, sc2, sc3, sc4 = st.columns(4)

with sc1:
    st.markdown(
        f"""
        <div class="dd-stat">
            <div class="dd-stat-label">Plant</div>
            <div class="dd-stat-value">{plant}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sc2:
    st.markdown(
        f"""
        <div class="dd-stat">
            <div class="dd-stat-label">Disease</div>
            <div class="dd-stat-value">{disease}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sc3:
    st.markdown(
        f"""
        <div class="dd-stat">
            <div class="dd-stat-label">Confidence</div>
            <div class="dd-stat-value">{probability:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sc4:
    risk_label = _risk_badge_label.get(alert_level, "Unknown")
    st.markdown(
        f"""
        <div class="dd-stat">
            <div class="dd-stat-label">Risk Level</div>
            <div style="margin-top:0.25rem;">
                <span class="dd-risk-badge {alert_level}">{risk_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)


# ── 3. Top Predictions Table ──

st.markdown(
    '<div class="dd-section"><div class="dd-section-title">Top Predictions</div></div>',
    unsafe_allow_html=True,
)

# Show top 5 predictions only (avoid oversized tables)
top_n = min(5, len(all_results))
for rank, entry in enumerate(all_results[:top_n], start=1):
    p_plant, p_disease = entry["plant"], entry["disease"]
    p_prob = entry["probability"]
    display_label = f"{p_plant} — {p_disease}"
    prob_display = f"{p_prob:.1%}" if p_prob >= 0.001 else "<0.1%"

    st.markdown(
        f"""
        <div class="dd-pred-row">
            <span class="dd-pred-rank">#{rank}</span>
            <span class="dd-pred-label">{display_label}</span>
            <span class="dd-pred-prob">{prob_display}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 4. Cleaned Image Preview (Expander) ──

with st.expander("Autoencoder Cleaned Image Preview"):
    cleaned_img_arr = result.get("cleaned_image")
    original_preview_arr = result.get("original_preview")

    if cleaned_img_arr is not None:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown(
                '<div class="dd-preview-label">Original (Resized)</div>',
                unsafe_allow_html=True,
            )
            if original_preview_arr is not None:
                st.image(
                    Image.fromarray(original_preview_arr),
                    use_container_width=True,
                )

        with img_col2:
            st.markdown(
                '<div class="dd-preview-label">After Autoencoder</div>',
                unsafe_allow_html=True,
            )
            st.image(
                Image.fromarray(cleaned_img_arr),
                use_container_width=True,
            )
    else:
        st.caption("Cleaned image data is not available.")


# ── 5. Probability Distribution Chart ──

st.markdown(
    '<div class="dd-section"><div class="dd-section-title">Probability Distribution</div></div>',
    unsafe_allow_html=True,
)

# Show top 10 predictions with > 0 probability for the chart
chart_results = [r for r in all_results if r["probability"] > 0][:10]
if len(chart_results) < 3:
    chart_results = all_results[:5]

# Reverse for horizontal bar chart (top prediction on top)
chart_results_rev = list(reversed(chart_results))

chart_labels = [f"{r['plant']} — {r['disease']}" for r in chart_results_rev]
chart_values = [r["probability"] * 100 for r in chart_results_rev]

# Highlight the top prediction
top_raw = result.get("top_disease_raw", "")
bar_colors = []
for r in chart_results_rev:
    if r["raw_label"] == top_raw:
        bar_colors.append("#40916C")
    else:
        bar_colors.append("#D5DDD9")

fig = go.Figure(
    go.Bar(
        x=chart_values,
        y=chart_labels,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(width=0),
            cornerradius=4,
        ),
        text=[f"{v:.1f}%" for v in chart_values],
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
    height=max(240, len(chart_results) * 36 + 60),
    margin=dict(l=10, r=50, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=12, color="#1A1F1C"),
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, max(chart_values) * 1.25] if chart_values else [0, 100],
    ),
    yaxis=dict(
        showgrid=False,
        tickfont=dict(family="DM Sans, sans-serif", size=11, color="#5A6B60"),
        automargin=True,
    ),
    bargap=0.25,
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── 6. Disclaimer Note ──

st.markdown(
    f"""
    <div class="dd-note">
        <div class="dd-note-title">Diagnostic Note</div>
        <p class="dd-note-text">
            This analysis was performed using a convolutional neural network (CNN)
            trained on the PlantVillage dataset, with an autoencoder-based image
            denoiser applied as a preprocessing step. The model identified
            <strong>{plant} — {disease}</strong> with {probability:.1%} confidence.
            For critical agricultural decisions, always cross-reference with
            local plant pathology expertise and laboratory diagnostics.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
