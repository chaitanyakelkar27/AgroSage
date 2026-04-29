"""
AgroSage - Agronomy Assistant
LangChain-powered Q&A interface.
"""

import streamlit as st
from langchain_groq import ChatGroq

from config.settings import get_env, get_llm_model
from utils import load_css, render_sidebar
from utils.qa_assistant import build_messages


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Agronomy Assistant — AgroSage",
    page_icon="Ag",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
render_sidebar()

# ── Page-specific styles ──────────────────────────────────────────────────────
st.markdown("""
<style>
.qa-note {
    background: var(--color-surface-alt);
    border: 1px solid var(--color-border-light);
    border-left: 3px solid var(--color-accent);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    margin-top: 0.75rem;
}
.qa-note-title {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}
.qa-note-text {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--color-text-secondary);
    line-height: 1.5;
    margin: 0;
}

/* Chat input readability */
[data-testid="stChatInput"] textarea {
    background: var(--color-surface) !important;
    color: var(--color-text) !important;
    border: 1.5px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #000000 !important;
    opacity: 1 !important;
}

/* Selectbox and dropdown readability */
[data-testid="stSelectbox"] input,
[data-testid="stSelectbox"] [role="combobox"] {
    color: var(--color-text) !important;
}
[data-testid="stSelectbox"] input::placeholder {
    color: #000000 !important;
    opacity: 1 !important;
}
[data-testid="stSelectbox"] [role="listbox"] * {
    color: var(--color-text) !important;
}

/* Chat message readability */
[data-testid="stChatMessage"] {
    background: var(--color-surface) !important;
    border: 1px solid var(--color-border-light) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.65rem 0.85rem !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"] * {
    color: var(--color-text) !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div style="margin-bottom:0.25rem;">
        <span class="ag-badge ag-badge-info" style="margin-bottom:0.5rem;">LangChain</span>
        <h1 style="font-size:1.5rem !important; font-weight:700 !important;
                   margin:0.35rem 0 0 0 !important; padding:0 !important;">
            Agronomy Assistant
        </h1>
        <p style="font-size:0.84rem; color:var(--color-text-secondary);
                  margin:0.15rem 0 0 0; font-weight:400;">
            Ask agronomy questions, clarify AgroSage workflows, and get guidance
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
api_key = get_env("GROQ_API_KEY")
configured_model = get_llm_model()

GROQ_MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

@st.cache_resource(show_spinner=False)
def _get_llm(model: str, temperature: float, key: str) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature, groq_api_key=key)


if "qa_messages" not in st.session_state:
    st.session_state["qa_messages"] = []


# ═══════════════════════════════════════════════════════════════════════════════
#  CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
control_col, settings_col = st.columns([2, 1])

with control_col:
    with st.container(border=True):
        focus = st.selectbox(
            "Focus area",
            options=[
                "General agronomy",
                "Crop recommendation",
                "Disease detection",
                "Weather and climate",
                "AgroSage app help",
            ],
            help="Pick the topic area to guide the assistant's response.",
        )

        if st.button("Reset chat", use_container_width=True):
            st.session_state["qa_messages"] = []
            st.rerun()

with settings_col:
    with st.container(border=True):
        temperature = st.slider(
            "Response creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Lower values are more factual; higher values are more creative.",
        )
        if configured_model not in GROQ_MODELS:
            st.caption(
                "Configured model is not in the supported list. "
                "Using a recommended Groq model instead."
            )
        model_name = st.selectbox(
            "Model",
            options=GROQ_MODELS,
            index=GROQ_MODELS.index(configured_model)
            if configured_model in GROQ_MODELS
            else 0,
            help="Pick a Groq model that is currently supported.",
        )
        st.markdown(
            f"<p style='margin:0; font-size:0.75rem; color:var(--color-text-muted);'>"
            f"Model: <strong>{model_name}</strong></p>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MESSAGES
# ═══════════════════════════════════════════════════════════════════════════════
for message in st.session_state["qa_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ═══════════════════════════════════════════════════════════════════════════════
#  INPUT
# ═══════════════════════════════════════════════════════════════════════════════
if not api_key:
    st.warning(
        "Add `GROQ_API_KEY` to your `.env` file to enable the assistant. "
        "See `.env.example` for a template."
    )
    st.stop()

user_prompt = st.chat_input(
    placeholder="Ask an agronomy question or request guidance...",
)

if user_prompt:
    st.session_state["qa_messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            messages = build_messages(user_prompt, focus)
            try:
                llm = _get_llm(model_name, temperature, api_key)
                response = llm.invoke(messages)
                answer = response.content.strip()
                st.markdown(answer)
            except Exception as exc:  # noqa: BLE001
                st.error(
                    "The selected Groq model is unavailable. "
                    "Pick another model or update `AGROSAGE_LLM_MODEL`."
                )
                st.caption(f"Details: {type(exc).__name__}: {exc}")
                answer = ""

    if answer:
        st.session_state["qa_messages"].append({"role": "assistant", "content": answer})

st.markdown(
    """
    <div class="qa-note">
        <div class="qa-note-title">Tip</div>
        <p class="qa-note-text">
            Include crop type, growth stage, and location to get more actionable guidance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
