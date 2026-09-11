"""Streamlit web dashboard for the Meeting Intelligence System."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

# ── Ensure project root is on the path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import configure_logging
from src.config import settings

configure_logging()
settings.warn_if_keys_missing()


def _save_keys_to_env(nvidia_key: str | None, hf_token: str | None) -> None:
    """Helper to persist user-entered keys to .env file."""
    env_path = Path(".env")
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated_nv = False
    updated_hf = False
    new_lines = []
    for line in lines:
        if line.startswith("NVIDIA_API_KEY="):
            new_lines.append(f"NVIDIA_API_KEY={nvidia_key or ''}")
            updated_nv = True
        elif line.startswith("HUGGINGFACE_TOKEN="):
            new_lines.append(f"HUGGINGFACE_TOKEN={hf_token or ''}")
            updated_hf = True
        else:
            new_lines.append(line)

    if not updated_nv and nvidia_key:
        new_lines.append(f"NVIDIA_API_KEY={nvidia_key}")
    if not updated_hf and hf_token:
        new_lines.append(f"HUGGINGFACE_TOKEN={hf_token}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Intelligence System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Pure Black theme with neon accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e0e0e0;
  }
  .stApp {
    background: #000000 !important;
    color: #e0e0e0;
  }
  header[data-testid="stHeader"],
  [data-testid="stHeader"],
  .stAppHeader,
  [data-testid="stToolbar"] {
    background: #000000 !important;
    color: #e0e0e0 !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0a0a0a !important;
    border-right: 1px solid #1a1a1a !important;
  }
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p {
    color: #cccccc !important;
  }

  /* ── Main content area ── */
  .main .block-container {
    background: #000000;
    padding-top: 1.5rem;
  }

  /* ── Headings ── */
  h1, h2, h3, h4 { color: #ffffff !important; }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 10px;
    padding: 14px;
  }
  [data-testid="metric-container"] label {
    color: #00ff88 !important;
    font-size: 0.75em !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6em !important;
    font-weight: 700 !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #0a0a0a !important;
    border-bottom: 1px solid #1a1a1a !important;
    gap: 2px;
    padding: 4px;
    border-radius: 8px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #666666 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.87em !important;
    padding: 6px 14px !important;
    border: none !important;
    transition: all 0.2s !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #00ff88 !important;
    background: #111111 !important;
  }
  .stTabs [aria-selected="true"] {
    background: #0d1a0d !important;
    color: #00ff88 !important;
    border: 1px solid #00ff8833 !important;
  }

  /* ── Buttons (Secondary / Default) ── */
  .stButton > button,
  button[data-testid="stBaseButton-secondary"],
  button[data-testid="baseButton-secondary"] {
    background: #000000 !important;
    color: #00ff88 !important;
    border: 1px solid #00ff88 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9em !important;
    transition: all 0.2s !important;
    letter-spacing: 0.03em !important;
  }
  .stButton > button *,
  button[data-testid="stBaseButton-secondary"] *,
  button[data-testid="baseButton-secondary"] * {
    color: #00ff88 !important;
  }
  .stButton > button:hover,
  button[data-testid="stBaseButton-secondary"]:hover,
  button[data-testid="baseButton-secondary"]:hover {
    background: #00ff88 !important;
    color: #000000 !important;
    box-shadow: 0 0 20px #00ff8844 !important;
  }
  .stButton > button:hover *,
  button[data-testid="stBaseButton-secondary"]:hover *,
  button[data-testid="baseButton-secondary"]:hover * {
    color: #000000 !important;
  }

  /* ── Buttons (Primary) ── */
  .stButton > button[kind="primary"],
  button[kind="primary"],
  button[data-testid="stBaseButton-primary"],
  button[data-testid="baseButton-primary"] {
    background: #00ff88 !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 0 15px rgba(0, 255, 136, 0.35) !important;
  }
  .stButton > button[kind="primary"] *,
  button[kind="primary"] *,
  button[data-testid="stBaseButton-primary"] *,
  button[data-testid="baseButton-primary"] *,
  .stButton > button[kind="primary"] p,
  button[kind="primary"] p,
  button[data-testid="stBaseButton-primary"] p {
    color: #000000 !important;
    font-weight: 700 !important;
  }
  .stButton > button[kind="primary"]:hover,
  button[kind="primary"]:hover,
  button[data-testid="stBaseButton-primary"]:hover,
  button[data-testid="baseButton-primary"]:hover {
    background: #00e077 !important;
    box-shadow: 0 0 30px #00ff8877 !important;
  }
  .stButton > button[kind="primary"]:hover *,
  button[kind="primary"]:hover *,
  button[data-testid="stBaseButton-primary"]:hover *,
  button[data-testid="baseButton-primary"]:hover * {
    color: #000000 !important;
  }
  .stButton > button[kind="primary"]:disabled,
  button[kind="primary"]:disabled,
  button[data-testid="stBaseButton-primary"]:disabled,
  button[data-testid="baseButton-primary"]:disabled {
    background: #1a402d !important;
    border: 1px solid #23543b !important;
    opacity: 0.7 !important;
  }
  .stButton > button[kind="primary"]:disabled *,
  button[kind="primary"]:disabled *,
  button[data-testid="stBaseButton-primary"]:disabled *,
  button[data-testid="baseButton-primary"]:disabled * {
    color: #00ff88 !important;
    font-weight: 600 !important;
  }

  /* ── Download buttons ── */
  .stDownloadButton > button,
  .stDownloadButton button {
    background: #000000 !important;
    color: #00aaff !important;
    border: 1px solid #00aaff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
  }
  .stDownloadButton > button *,
  .stDownloadButton button * {
    color: #00aaff !important;
  }
  .stDownloadButton > button:hover,
  .stDownloadButton button:hover {
    background: #00aaff !important;
    color: #000000 !important;
    box-shadow: 0 0 20px #00aaff44 !important;
  }
  .stDownloadButton > button:hover *,
  .stDownloadButton button:hover * {
    color: #000000 !important;
  }

  /* ── Progress bar ── */
  .stProgress > div > div {
    background: linear-gradient(90deg, #00ff88, #00aaff) !important;
    border-radius: 4px;
    box-shadow: 0 0 8px #00ff8866;
  }
  .stProgress > div {
    background: #111111 !important;
    border-radius: 4px;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploaderDropzone"] {
    background: #050505 !important;
    border: 2px dashed #00ff8844 !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #00ff88 !important;
    box-shadow: 0 0 20px #00ff8811 !important;
  }

  /* ── Selects & inputs ── */
  .stSelectbox > div > div,
  .stTextInput > div > div,
  .stTextArea > div > div,
  .stNumberInput > div > div {
    background: #0a0a0a !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
  }
  .stSelectbox > div > div:focus-within,
  .stTextInput > div > div:focus-within,
  .stTextArea > div > div:focus-within {
    border-color: #00ff8866 !important;
    box-shadow: 0 0 0 2px #00ff8811 !important;
  }

  /* ── Toggle ── */
  .stToggle [data-testid="stMarkdownContainer"] p { color: #cccccc !important; }

  /* ── DataFrames ── */
  .stDataFrame {
    border: 1px solid #1a1a1a !important;
    border-radius: 10px !important;
    overflow: hidden !important;
  }
  .stDataFrame th {
    background: #0d0d0d !important;
    color: #00ff88 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.78em !important;
    letter-spacing: 0.06em !important;
  }
  .stDataFrame td { background: #050505 !important; color: #d0d0d0 !important; }
  .stDataFrame tr:hover td { background: #0d0d0d !important; }

  /* ── Alerts ── */
  .stAlert { border-radius: 8px !important; }
  .stSuccess { background: #001a0d !important; border-left: 3px solid #00ff88 !important; }
  .stWarning { background: #1a1200 !important; border-left: 3px solid #ffaa00 !important; }
  .stError { background: #1a0000 !important; border-left: 3px solid #ff4444 !important; }
  .stInfo { background: #00101a !important; border-left: 3px solid #00aaff !important; }

  /* ── Expanders ── */
  .streamlit-expanderHeader {
    background: #0a0a0a !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
    color: #cccccc !important;
  }
  .streamlit-expanderContent {
    background: #050505 !important;
    border: 1px solid #1a1a1a !important;
    border-top: none !important;
  }

  /* ── Dividers ── */
  hr { border-color: #1a1a1a !important; }

  /* ── Transcript segment cards ── */
  .tseg {
    background: #080808;
    border-left: 3px solid #00ff88;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 3px 0;
    font-family: 'Inter', sans-serif;
    transition: background 0.15s;
  }
  .tseg:hover { background: #0f0f0f; }
  .tseg .ts {
    color: #00ff88;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74em;
    font-weight: 600;
    letter-spacing: 0.05em;
  }
  .tseg .sp {
    font-size: 0.82em;
    font-weight: 600;
    margin: 0 4px;
  }
  .tseg .tx { color: #d0d0d0; font-size: 0.93em; }

  /* Speaker colours */
  .sp-0 { color: #00ff88; }
  .sp-1 { color: #00aaff; }
  .sp-2 { color: #ff6b9d; }
  .sp-3 { color: #ffaa00; }
  .sp-4 { color: #cc88ff; }
  .sp-5 { color: #ff8844; }
  .sp-none { color: #666666; }

  /* ── Search highlight ── */
  .hl { background: #00ff8822; border-radius: 3px; padding: 1px 3px; color: #00ff88; font-weight: 600; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #080808; }
  ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #3a3a3a; }

  /* ── Caption / small text ── */
  .stCaption, small, .caption { color: #555555 !important; }

  /* ── Sidebar section headers ── */
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #00ff88 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in {"result": None, "processing": False, "error": None}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ Meeting Intelligence")
    st.markdown("<small style='color:#555;'>Powered by Llama + Whisper</small>", unsafe_allow_html=True)
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Meeting Audio",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
        help="Supported: WAV, MP3, M4A, FLAC, OGG, AAC",
        key="audio_uploader",
    )

    # ── User API Key Configuration ───────────────────────────────────────────
    with st.expander(
        "🔑 Configure API Keys & Tokens",
        expanded=not (bool(settings.nvidia_api_key) and bool(settings.huggingface_token)),
    ):
        st.caption("Provide API credentials here to unlock AI Analysis & Diarization without touching config files.")

        user_nv_key = st.text_input(
            "NVIDIA API Key",
            value=st.session_state.get("user_nv_key", settings.nvidia_api_key or ""),
            type="password",
            placeholder="nvapi-...",
            help="Free API key from https://build.nvidia.com",
            key="cfg_nv_key",
        )
        if user_nv_key.strip() != (settings.nvidia_api_key or ""):
            settings.nvidia_api_key = user_nv_key.strip() or None
            st.session_state.user_nv_key = settings.nvidia_api_key or ""
            st.rerun()

        user_hf_token = st.text_input(
            "HuggingFace Token",
            value=st.session_state.get("user_hf_token", settings.huggingface_token or ""),
            type="password",
            placeholder="hf_...",
            help="Token from https://huggingface.co/settings/tokens (requires accepting pyannote conditions)",
            key="cfg_hf_token",
        )
        if user_hf_token.strip() != (settings.huggingface_token or ""):
            settings.huggingface_token = user_hf_token.strip() or None
            st.session_state.user_hf_token = settings.huggingface_token or ""
            if settings.huggingface_token:
                os.environ["HUGGINGFACE_TOKEN"] = settings.huggingface_token
                os.environ["HF_TOKEN"] = settings.huggingface_token
            st.rerun()

        if st.button("💾 Save Keys to .env", use_container_width=True, key="save_keys_btn"):
            _save_keys_to_env(settings.nvidia_api_key, settings.huggingface_token)
            st.success("Keys saved to .env!")

    st.markdown("### ⚙️ Processing Options")

    language_options = {
        "Auto-detect": None,
        "English": "en",
        "Hindi": "hi",
        "Marathi": "mr",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Arabic": "ar",
        "Portuguese": "pt",
    }
    selected_lang = st.selectbox("Language", list(language_options.keys()), index=0)

    model_options = {
        "Llama 3.2 11B (Fast & Recommended)": "meta/llama-3.2-11b-vision-instruct",
        "Llama 3.2 90B (High Quality)": "meta/llama-3.2-90b-vision-instruct",
        "Mistral Large 2": "mistralai/mistral-large-2-instruct",
    }
    # Only show LLM model selector if NVIDIA key is set
    if settings.nvidia_api_key:
        selected_llm_label = st.selectbox(
            "LLM Model (NVIDIA NIM)",
            list(model_options.keys()),
            index=0,
            help="Llama 3.2 11B is fast and recommended for structured extraction.",
        )
        selected_llm = model_options[selected_llm_label]
    else:
        selected_llm = settings.openai_model

    whisper_size = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=1,
        help="base = fast/good. small = balanced. large-v3 = best accuracy.",
    )

    enable_vad = st.toggle("Voice Activity Detection (VAD)", value=True)
    enable_diarization = st.toggle(
        "Speaker Diarization",
        value=settings.diarization_available,
        disabled=not settings.diarization_available,
        help="Requires HUGGINGFACE_TOKEN." if not settings.diarization_available else "Who said what.",
    )
    enable_llm = st.toggle(
        "AI Analysis (Summary + Actions)",
        value=settings.llm_available,
        disabled=not settings.llm_available,
        help="Requires NVIDIA_API_KEY." if not settings.llm_available else "Generate meeting insights.",
    )

    num_speakers = None
    if enable_diarization:
        ns = st.number_input("Speakers (0 = auto)", min_value=0, max_value=20, value=0)
        if ns > 0:
            num_speakers = ns

    st.divider()

    # API status panel
    st.markdown("### 🔑 API Status")
    nvidia_ok = bool(settings.nvidia_api_key)
    openai_ok = bool(settings.openai_api_key)
    dia_ok = settings.diarization_available

    if nvidia_ok:
        st.markdown("✅ **NVIDIA API** (Llama)")
    elif openai_ok:
        st.markdown("✅ **OpenAI API** (fallback)")
    else:
        st.markdown("❌ **No LLM key set**")
        st.caption("Enter in 'Configure API Keys' above or add to .env")

    st.markdown(f"{'✅' if dia_ok else '❌'} **HuggingFace** (diarization)")
    if not dia_ok:
        st.caption("Enter in 'Configure API Keys' above or add to .env")

    st.divider()

    process_btn = st.button(
        "▶  Process Meeting",
        use_container_width=True,
        disabled=uploaded_file is None or st.session_state.processing,
        type="primary",
        key="process_btn",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────
if process_btn and uploaded_file is not None:
    st.session_state.processing = True
    st.session_state.result = None
    st.session_state.error = None

    # Override model settings
    settings.whisper_model_size = whisper_size
    settings.openai_model = selected_llm

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    prog_col, stat_col = st.columns([3, 2])
    with prog_col:
        progress_bar = st.progress(0)
    with stat_col:
        status_text = st.empty()

    def _update_progress(msg: str, pct: int) -> None:
        progress_bar.progress(pct)
        status_text.markdown(
            f"<span style='color:#00ff88; font-size:0.85em; font-weight:600;'>"
            f"⟳ {msg}</span>",
            unsafe_allow_html=True,
        )

    try:
        from src.pipeline import process_meeting
        result = process_meeting(
            file_path=tmp_path,
            language=language_options[selected_lang],
            enable_vad=enable_vad,
            enable_diarization=enable_diarization,
            enable_llm=enable_llm,
            num_speakers=num_speakers,
            progress_callback=_update_progress,
        )
        st.session_state.result = result
        progress_bar.progress(100)
        status_text.markdown(
            "<span style='color:#00ff88; font-weight:700;'>✓ Complete</span>",
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.session_state.error = str(exc)
        status_text.markdown(
            f"<span style='color:#ff4444; font-weight:600;'>✗ {exc}</span>",
            unsafe_allow_html=True,
        )
    finally:
        st.session_state.processing = False
        try:
            tmp_path.unlink()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(f"**Processing Error:** {st.session_state.error}")


# ─────────────────────────────────────────────────────────────────────────────
# Hero / Landing state
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.result is None and not st.session_state.error:
    st.markdown("""
    <div style='text-align:center; padding:60px 0 40px 0;'>
      <div style='font-size:3.2em; margin-bottom:8px;'>🎙️</div>
      <h1 style='font-size:2.6em; color:#ffffff; margin:0 0 8px 0; font-weight:700;'>
        Meeting Intelligence
      </h1>
      <p style='color:#00ff88; font-size:1.1em; font-weight:500; margin:0 0 4px 0;'>
        Powered by Llama 3.1 &nbsp;·&nbsp; faster-whisper &nbsp;·&nbsp; pyannote
      </p>
      <p style='color:#444; font-size:0.95em; margin-top:12px;'>
        Upload a meeting recording → get a timestamped transcript,<br>
        speaker attribution, AI summary, decisions, and action items.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    card_style = "background:#080808; border:1px solid #1a1a1a; border-radius:10px; padding:20px; text-align:center;"
    with c1:
        st.markdown(f"<div style='{card_style}'><div style='font-size:1.8em'>🎤</div><div style='color:#00ff88; font-weight:600; margin:6px 0 4px;'>Speech-to-Text</div><div style='color:#555; font-size:0.82em;'>Whisper ASR<br>Auto language detection</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='{card_style}'><div style='font-size:1.8em'>👥</div><div style='color:#00aaff; font-weight:600; margin:6px 0 4px;'>Speaker ID</div><div style='color:#555; font-size:0.82em;'>Who said what<br>& when</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div style='{card_style}'><div style='font-size:1.8em'>🤖</div><div style='color:#ff6b9d; font-weight:600; margin:6px 0 4px;'>AI Analysis</div><div style='color:#555; font-size:0.82em;'>Llama 3.1 summaries<br>& action items</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div style='{card_style}'><div style='font-size:1.8em'>📤</div><div style='color:#ffaa00; font-weight:600; margin:6px 0 4px;'>Export</div><div style='color:#555; font-size:0.82em;'>JSON · TXT<br>Markdown reports</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#333; font-size:0.85em;'>"
        "← Upload an audio file from the sidebar to begin"
        "</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.result is not None:
    result = st.session_state.result
    meta = result.audio_metadata
    summary = result.summary

    # ── Speaker colour map ────────────────────────────────────────────────────
    SPEAKER_COLORS = ["#00ff88", "#00aaff", "#ff6b9d", "#ffaa00", "#cc88ff", "#ff8844"]
    SP_CLASSES = ["sp-0", "sp-1", "sp-2", "sp-3", "sp-4", "sp-5"]
    unique_speakers = sorted(set(s.speaker for s in result.aligned_transcript if s.speaker))
    sp_color = {sp: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, sp in enumerate(unique_speakers)}
    sp_class = {sp: SP_CLASSES[i % len(SP_CLASSES)] for i, sp in enumerate(unique_speakers)}

    # ── Top metrics bar ───────────────────────────────────────────────────────
    st.markdown("<div style='margin-bottom:4px;'>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Duration", meta.duration_formatted)
    c2.metric("Language", result.transcription.language.upper())
    c3.metric("Segments", len(result.aligned_transcript))
    c4.metric("Words", f"{len(result.transcription.full_text.split()):,}")
    if result.processing_time_seconds and meta.duration_seconds:
        rtf = result.processing_time_seconds / meta.duration_seconds
        c5.metric("RTF", f"{rtf:.2f}x", help="Real-Time Factor: <1 = faster than real-time")
    else:
        c5.metric("RTF", "—")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📋 Overview",
        "📝 Transcript",
        "💡 Key Points",
        "✅ Decisions",
        "📌 Action Items",
        "🔍 Search",
        "📈 Evaluation",
        "💾 Export",
    ])

    # ─── TAB 0: Overview ──────────────────────────────────────────────────────
    with tabs[0]:
        # ── Determine if AI summary is populated ──────────────────────────────
        has_ai_summary = bool(
            summary and (summary.summary.strip() or summary.key_points or summary.topics)
        )

        # ── Show any LLM error banner ─────────────────────────────────────────
        if getattr(result, "llm_error", None):
            st.warning(f"⚠️ **AI Analysis error:** {result.llm_error}")

        # ── AI Summary section ────────────────────────────────────────────────
        if has_ai_summary:
            if summary.title:
                st.markdown(
                    f"<h2 style='color:#00ff88; margin-top:0;'>📅 {summary.title}</h2>",
                    unsafe_allow_html=True,
                )

            if summary.participants:
                pills = " ".join(
                    f"<span style='background:#0d1a0d; border:1px solid #00ff8833; "
                    f"color:#00ff88; padding:3px 10px; border-radius:20px; "
                    f"font-size:0.82em; margin:2px; display:inline-block;'>{p}</span>"
                    for p in summary.participants
                )
                st.markdown(f"<div style='margin-bottom:12px;'>{pills}</div>", unsafe_allow_html=True)
            elif unique_speakers:
                # Fall back to detected speakers from transcript
                pills = " ".join(
                    f"<span style='background:#0d0d1a; border:1px solid #00aaff33; "
                    f"color:{sp_color.get(s, '#00aaff')}; padding:3px 10px; border-radius:20px; "
                    f"font-size:0.82em; margin:2px; display:inline-block;'>{s}</span>"
                    for s in unique_speakers
                )
                st.markdown(f"<div style='margin-bottom:12px;'>{pills}</div>", unsafe_allow_html=True)

            if summary.summary.strip():
                st.markdown(
                    f"<div style='background:#050505; border-left:3px solid #00ff88; "
                    f"border-radius:0 8px 8px 0; padding:14px 18px; color:#cccccc; "
                    f"font-size:0.95em; line-height:1.7;'>{summary.summary}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("ℹ️ AI summary text was not returned by the model. Try clicking **Regenerate AI Summary** below.")

            if summary.topics:
                st.markdown("<br>**Topics discussed:**", unsafe_allow_html=True)
                cols = st.columns(min(len(summary.topics), 4))
                for i, t in enumerate(summary.topics):
                    if not t.topic.strip():
                        continue
                    with cols[i % 4]:
                        timing = f"<br><span style='color:#555; font-size:0.78em;'>{t.start or ''}{'–' if t.start and t.end else ''}{t.end or ''}</span>" if t.start or t.end else ""
                        st.markdown(
                            f"<div style='background:#080808; border:1px solid #1a1a1a; "
                            f"border-radius:8px; padding:12px; text-align:center;'>"
                            f"<span style='color:#ffffff; font-weight:600;'>{t.topic}</span>"
                            f"{timing}</div>",
                            unsafe_allow_html=True,
                        )

        else:
            # ── No AI summary yet ─────────────────────────────────────────────
            # Show speaker pills from transcript even without AI
            if unique_speakers:
                pills = " ".join(
                    f"<span style='background:#0d0d1a; border:1px solid #00aaff33; "
                    f"color:{sp_color.get(s, '#00aaff')}; padding:3px 10px; border-radius:20px; "
                    f"font-size:0.82em; margin:2px; display:inline-block;'>{s}</span>"
                    for s in unique_speakers
                )
                st.markdown(
                    f"<div style='margin-bottom:14px;'><span style='color:#555; font-size:0.8em;'>Speakers detected: </span>{pills}</div>",
                    unsafe_allow_html=True,
                )

            if not getattr(result, "llm_error", None):
                st.info("💡 **AI Analysis** is not enabled or did not run. Click **⚡ Generate AI Summary Now** below.")

        # ── Regenerate button (always visible when no complete summary) ───────
        if not has_ai_summary or (summary and not summary.summary.strip()):
            if st.button("⚡ Generate AI Summary Now", key="retry_llm_btn", type="primary"):
                with st.spinner("Generating AI analysis with Llama 3.2..."):
                    try:
                        from src.meeting.summarizer import analyze_meeting
                        settings.openai_model = selected_llm
                        new_summary = analyze_meeting(result.aligned_transcript)
                        result.summary = new_summary
                        result.llm_error = None
                        st.session_state.result = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to generate summary: {e}")

        # ── Transcript Preview (always visible) ───────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<span style='color:#555; font-size:0.8em; font-weight:600; letter-spacing:0.05em;'>TRANSCRIPT PREVIEW</span>",
            unsafe_allow_html=True,
        )
        preview_segs = result.aligned_transcript[:10] if result.aligned_transcript else []
        if preview_segs:
            for seg in preview_segs:
                color = sp_color.get(seg.speaker or "", "#666666")
                sp_label = seg.speaker or "Speaker"
                st.markdown(
                    f'<div class="tseg"><span class="ts">{seg.timestamp_str}</span> '
                    f'<span class="sp" style="color:{color};">{sp_label}</span>'
                    f'<span class="tx"> {seg.text}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No transcript segments available.")

    # ─── TAB 1: Transcript ────────────────────────────────────────────────────
    with tabs[1]:
        c_left, c_right = st.columns([4, 1])
        with c_left:
            st.markdown(
                f"<h3 style='margin:0; color:#ffffff;'>Full Transcript</h3>"
                f"<span style='color:#444; font-size:0.82em;'>"
                f"{len(result.aligned_transcript)} segments · "
                f"{len(result.transcription.full_text.split()):,} words · "
                f"{result.transcription.language.upper()}</span>",
                unsafe_allow_html=True,
            )
        with c_right:
            if unique_speakers:
                for sp in unique_speakers:
                    color = sp_color.get(sp, "#666")
                    st.markdown(
                        f"<span style='color:{color}; font-size:0.8em; font-weight:600;'>● {sp}</span>",
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        for seg in result.aligned_transcript:
            color = sp_color.get(seg.speaker or "", "#555555")
            sp_label = seg.speaker or "Unknown"
            st.markdown(
                f'<div class="tseg" style="border-left-color:{color};">'
                f'<span class="ts">{seg.timestamp_str}</span>'
                f'<span class="sp" style="color:{color};"> {sp_label}</span>'
                f'<span class="tx"> {seg.text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ─── TAB 2: Key Points ────────────────────────────────────────────────────
    with tabs[2]:
        display_key_points = []
        if summary and summary.key_points:
            display_key_points = summary.key_points
        elif summary and summary.summary:
            from src.meeting.summarizer import extract_fallback_key_points
            display_key_points = extract_fallback_key_points(summary.summary)
            if display_key_points and not summary.key_points:
                summary.key_points = display_key_points

        if display_key_points:
            st.markdown("### 💡 Key Discussion Points")
            for i, pt in enumerate(display_key_points, 1):
                st.markdown(
                    f"<div style='background:#080808; border:1px solid #1a1a1a; "
                    f"border-radius:8px; padding:12px 16px; margin:6px 0; "
                    f"display:flex; align-items:center; gap:12px;'>"
                    f"<span style='color:#00ff88; font-weight:700; font-size:1.1em; "
                    f"min-width:28px; text-align:center;'>{i:02d}</span>"
                    f"<span style='color:#d0d0d0;'>{pt}</span></div>",
                    unsafe_allow_html=True,
                )
        elif summary:
            st.info("ℹ️ No specific key points were extracted for this meeting.")
            if st.button("⚡ Extract Key Points Now", key="retry_kp_btn"):
                with st.spinner("Extracting key points with AI..."):
                    try:
                        from src.meeting.summarizer import analyze_meeting
                        settings.openai_model = selected_llm
                        new_summary = analyze_meeting(result.aligned_transcript)
                        result.summary = new_summary
                        st.session_state.result = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("💡 AI Analysis is not enabled. Enable AI Analysis in the sidebar or click below to generate.")
            if st.button("⚡ Generate AI Summary Now", key="generate_ai_kp_btn", type="primary"):
                with st.spinner("Generating AI analysis with Llama 3.2..."):
                    try:
                        from src.meeting.summarizer import analyze_meeting
                        settings.openai_model = selected_llm
                        new_summary = analyze_meeting(result.aligned_transcript)
                        result.summary = new_summary
                        result.llm_error = None
                        st.session_state.result = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ─── TAB 3: Decisions ─────────────────────────────────────────────────────
    with tabs[3]:
        if summary and summary.decisions:
            st.markdown("### ✅ Decisions Made")
            for d in summary.decisions:
                ts_badge = f"<span style='background:#001a1a; color:#00aaff; border:1px solid #00aaff33; border-radius:20px; padding:2px 8px; font-size:0.75em; font-family:monospace; margin-left:8px;'>⏱ {d.timestamp}</span>" if d.timestamp else ""
                ev = f"<div style='color:#555; font-size:0.85em; margin-top:8px; font-style:italic;'>\"{d.evidence}\"</div>" if d.evidence else ""
                st.markdown(
                    f"<div style='background:#080808; border:1px solid #1a1a1a; "
                    f"border-radius:10px; padding:14px 18px; margin:8px 0;'>"
                    f"<span style='color:#ffffff; font-weight:600;'>📌 {d.decision}</span>"
                    f"{ts_badge}{ev}</div>",
                    unsafe_allow_html=True,
                )
            if summary.open_questions:
                st.markdown("### ❓ Open Questions")
                for q in summary.open_questions:
                    st.markdown(
                        f"<div style='background:#080808; border-left:3px solid #ffaa00; "
                        f"border-radius:0 8px 8px 0; padding:10px 14px; margin:5px 0; color:#cccccc;'>{q}</div>",
                        unsafe_allow_html=True,
                    )
        elif summary:
            st.info("ℹ️ No formal decisions were explicitly finalized in this meeting.")
            if summary.open_questions:
                st.markdown("### ❓ Open Questions")
                for q in summary.open_questions:
                    st.markdown(
                        f"<div style='background:#080808; border-left:3px solid #ffaa00; "
                        f"border-radius:0 8px 8px 0; padding:10px 14px; margin:5px 0; color:#cccccc;'>{q}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("💡 AI Analysis was not enabled. Enable AI Analysis in the sidebar to extract decisions.")

    # ─── TAB 4: Action Items ──────────────────────────────────────────────────
    with tabs[4]:
        if summary and summary.action_items:
            st.markdown("### 📌 Action Items")
            df = pd.DataFrame([{
                "Owner": a.owner or "TBD",
                "Task": a.task,
                "Deadline": a.deadline or "—",
                "Timestamp": a.timestamp or "—",
                "Confidence": f"{a.confidence:.0%}" if a.confidence else "—",
            } for a in summary.action_items])
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("<br>**Details**", unsafe_allow_html=True)
            for a in summary.action_items:
                conf_color = "#00ff88" if (a.confidence or 0) > 0.7 else "#ffaa00" if (a.confidence or 0) > 0.4 else "#ff4444"
                with st.expander(f"📋 {a.task}"):
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.markdown(f"**Owner**\n\n{a.owner or 'TBD'}")
                    cc2.markdown(f"**Deadline**\n\n{a.deadline or 'Not specified'}")
                    cc3.markdown(f"**Timestamp**\n\n{a.timestamp or '—'}")
                    if a.confidence:
                        st.markdown(
                            f"Confidence: <span style='color:{conf_color}; font-weight:700;'>{a.confidence:.0%}</span>",
                            unsafe_allow_html=True,
                        )
                    if a.evidence:
                        st.markdown(
                            f"<div style='color:#555; font-style:italic; font-size:0.88em; "
                            f"margin-top:8px; border-left:2px solid #2a2a2a; padding-left:10px;'>"
                            f"\"{a.evidence}\"</div>",
                            unsafe_allow_html=True,
                        )
        elif summary:
            st.info("ℹ️ No specific action items or task assignments were detected in this meeting.")
        else:
            st.info("💡 AI Analysis was not enabled. Enable AI Analysis in the sidebar to extract action items.")

    # ─── TAB 5: Search ────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("### 🔍 Search Transcript")
        query = st.text_input(
            "",
            placeholder='Search for a word or phrase — e.g. "deployment" or "API"',
            key="search_q",
            label_visibility="collapsed",
        )
        if query:
            matches = result.search_transcript(query)
            if matches:
                st.markdown(
                    f"<div style='color:#00ff88; font-size:0.9em; margin-bottom:10px;'>"
                    f"Found <b>{len(matches)}</b> segment(s) matching &ldquo;<b>{query}</b>&rdquo;</div>",
                    unsafe_allow_html=True,
                )
                for seg in matches:
                    color = sp_color.get(seg.speaker or "", "#555")
                    sp_label = seg.speaker or "Unknown"
                    # Highlight matched text
                    hi_text = seg.text
                    for variant in [query, query.lower(), query.upper(), query.capitalize()]:
                        hi_text = hi_text.replace(
                            variant,
                            f'<span class="hl">{variant}</span>',
                        )
                    st.markdown(
                        f'<div class="tseg" style="border-left-color:{color};">'
                        f'<span class="ts">{seg.timestamp_str}</span>'
                        f'<span class="sp" style="color:{color};"> {sp_label}</span>'
                        f'<span class="tx"> {hi_text}</span></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"<div style='color:#ffaa00;'>No results for \"<b>{query}</b>\"</div>",
                    unsafe_allow_html=True,
                )

    # ─── TAB 6: Evaluation ────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("### 📈 Processing Metrics")
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Process Time", f"{result.processing_time_seconds:.1f}s" if result.processing_time_seconds else "—")
        if result.processing_time_seconds and meta.duration_seconds:
            rtf = result.processing_time_seconds / meta.duration_seconds
            ec2.metric("Real-Time Factor", f"{rtf:.3f}",
                       delta="faster than RT" if rtf < 1 else "slower than RT",
                       delta_color="normal" if rtf < 1 else "inverse")
        ec3.metric("Sample Rate", f"{meta.sample_rate:,} Hz")

        st.divider()
        st.markdown("### 📐 Word Error Rate (WER)")
        st.markdown("<span style='color:#555; font-size:0.87em;'>Paste a reference transcript to measure transcription accuracy.</span>", unsafe_allow_html=True)
        ref = st.text_area("Reference Transcript", placeholder="Paste ground-truth text here...", height=100, key="wer_ref", label_visibility="collapsed")
        if st.button("Calculate WER", key="calc_wer") and ref:
            from evaluation.wer import calculate_wer, wer_report
            try:
                wr = calculate_wer(ref, result.transcription.full_text)
                wc1, wc2, wc3, wc4 = st.columns(4)
                wc1.metric("WER", f"{wr.wer:.1%}")
                wc2.metric("Substitutions", wr.substitutions)
                wc3.metric("Deletions", wr.deletions)
                wc4.metric("Insertions", wr.insertions)
            except Exception as e:
                st.error(f"WER error: {e}")

    # ─── TAB 7: Export ────────────────────────────────────────────────────────
    with tabs[7]:
        st.markdown("### 💾 Export Meeting Report")
        from src.meeting.export import to_json, to_txt, to_markdown

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.markdown("<div style='color:#00ff88; font-weight:600; margin-bottom:8px;'>JSON</div>Full structured data", unsafe_allow_html=True)
            st.download_button("⬇ Download JSON", data=to_json(result),
                               file_name=f"{result.meeting_id}.json", mime="application/json",
                               use_container_width=True, key="dl_json")
        with dc2:
            st.markdown("<div style='color:#00aaff; font-weight:600; margin-bottom:8px;'>TXT</div>Plain text report", unsafe_allow_html=True)
            st.download_button("⬇ Download TXT", data=to_txt(result),
                               file_name=f"{result.meeting_id}.txt", mime="text/plain",
                               use_container_width=True, key="dl_txt")
        with dc3:
            st.markdown("<div style='color:#ff6b9d; font-weight:600; margin-bottom:8px;'>Markdown</div>Professional report", unsafe_allow_html=True)
            st.download_button("⬇ Download MD", data=to_markdown(result),
                               file_name=f"{result.meeting_id}.md", mime="text/markdown",
                               use_container_width=True, key="dl_md")

        st.divider()
        with st.expander("Preview Markdown Report"):
            st.markdown(to_markdown(result))
