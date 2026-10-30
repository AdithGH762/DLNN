"""EmotionSense Studio — Streamlit dashboard.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import io
import hashlib
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # must be before pyplot import; prevents GUI thread issues in Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── project imports ──────────────────────────────────────────────────────────
from config import MAX_FRAMES, MODEL_DIR, OUTPUT_DIR, PROJECT_ROOT, SAMPLE_RATE
from predict import (
    CNN_MAX_AUDIO_SECONDS,
    ORIGINAL_CNN_LSTM,
    SVM_BASELINE,
    available_prediction_models,
    missing_model_artifacts,
    predict_file,
)
from src.noise import add_white_noise_at_snr, derive_noise_seed
from src.provenance import input_key, run_provenance_path, validate_run_binding
from src.metrics import evaluation_samples_match
from src.dataset import (
    discover_ravdess,
    parse_ravdess_filename,
    validate_test_manifest_labels,
)

# ── page config must be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="EmotionSense Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS INJECTION
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
:root {
    color-scheme: dark;
    --bg-deep: #0b0e14;
    --bg-card: #121722;
    --bg-surface: #191f2b;
    --bg-raised: #202838;
    --accent: #9a8cff;
    --accent2: #7889f5;
    --accent-soft: rgba(154, 140, 255, 0.12);
    --success: #55c99a;
    --warning: #edbd67;
    --danger: #f1828c;
    --text-hi: #f3f4f8;
    --text-mid: #b0b8c8;
    --text-lo: #858fa2;
    --border: #2a3343;
    --radius-lg: 16px;
    --radius-md: 11px;
    --shadow-card: 0 12px 32px rgba(0, 0, 0, 0.18);
    --font-ui: Inter, "Segoe UI", Arial, sans-serif;
    --font-display: "Space Grotesk", Inter, "Segoe UI", Arial, sans-serif;
}

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg-deep) !important;
    font-family: var(--font-ui);
    color: var(--text-hi);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 82% 0%, rgba(91, 83, 163, 0.08), transparent 34rem),
        var(--bg-deep) !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1480px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #10151e !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
    background: transparent;
}
[data-testid="stHeader"] {
    background: rgba(11, 14, 20, 0.82) !important;
}

h1,
h2,
h3 {
    font-family: var(--font-display) !important;
    color: var(--text-hi);
    letter-spacing: -0.025em;
}
h1 {
    font-size: clamp(1.8rem, 3vw, 2.45rem) !important;
    line-height: 1.15 !important;
}
h2 {
    font-size: clamp(1.35rem, 2.2vw, 1.75rem) !important;
    line-height: 1.25 !important;
}
h3 {
    font-size: 1.08rem !important;
    color: var(--text-mid) !important;
    font-weight: 600 !important;
}

p,
[data-testid="stCaption"],
[data-testid="stMarkdownContainer"] {
    line-height: 1.62;
}

.hero-banner {
    background: linear-gradient(115deg, #151b27 0%, #171a2a 56%, #141b27 100%);
    border: 1px solid #30384a;
    border-radius: var(--radius-lg);
    padding: clamp(1.35rem, 3vw, 2.25rem) clamp(1.25rem, 3.5vw, 2.75rem);
    margin-bottom: 1.75rem;
    box-shadow: var(--shadow-card);
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-family: var(--font-display);
    font-size: clamp(2rem, 4vw, 2.8rem);
    font-weight: 700;
    color: #c4bbff;
    letter-spacing: -0.04em;
    line-height: 1.12;
    margin: 0 0 0.55rem;
}
.hero-sub {
    max-width: 70ch;
    color: var(--text-mid);
    font-size: 0.98rem;
    font-weight: 400;
    line-height: 1.65;
    margin: 0;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    margin-top: 1.2rem;
}
.status-ready {
    background: rgba(85, 201, 154, 0.1);
    color: #78dcb2;
    border: 1px solid rgba(85, 201, 154, 0.28);
}
.status-missing {
    background: rgba(241, 130, 140, 0.1);
    color: #f3a0a7;
    border: 1px solid rgba(241, 130, 140, 0.28);
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
}
.status-dot-pulse {
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.48; }
}

.es-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: clamp(1rem, 2vw, 1.45rem);
    margin-bottom: 1.05rem;
    box-shadow: var(--shadow-card);
    transition: border-color 160ms ease, transform 160ms ease;
}
.es-card:hover {
    border-color: #3a4559;
}
.es-card-title {
    font-family: var(--font-display);
    font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #b7adff; margin-bottom: 0.9rem;
}

[data-testid="stMetric"],
[data-testid="stMetricContainer"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.85rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-mid) !important;
    font-size: 0.76rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-hi) !important;
    font-size: 1.4rem !important;
    font-weight: 650 !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-surface) !important;
    border: 1px dashed #46516a !important;
    border-radius: var(--radius-lg) !important;
    transition: border-color 160ms ease, background-color 160ms ease;
}
[data-testid="stFileUploader"]:hover,
[data-testid="stFileUploader"]:focus-within {
    border-color: var(--accent) !important;
    background: #1c2331 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    color: var(--text-mid) !important;
    font-weight: 550;
    transition: color 140ms ease, background-color 140ms ease;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--text-hi) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

.es-sep {
    border-top: 1px solid var(--border);
    margin: 1.45rem 0;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 0.84rem;
}
.prob-label {
    width: 100px;
    color: var(--text-mid);
    text-align: right;
    text-transform: capitalize;
}
.prob-track {
    flex: 1;
    height: 7px;
    background: #0c1018;
    border-radius: 999px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.prob-pct {
    width: 48px;
    color: var(--text-hi);
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    font-size: 0.8rem;
}

.seg-row {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 7px;
    font-size: 0.83rem;
}
.seg-time {
    width: 80px;
    color: var(--text-mid);
    font-variant-numeric: tabular-nums;
}
.seg-label {
    padding: 4px 9px;
    border-radius: 6px;
    background: var(--bg-surface);
    color: var(--text-hi);
    font-weight: 600;
    border: 1px solid var(--border);
}
.seg-conf {
    color: var(--text-mid);
    font-size: 0.78rem;
}

.setup-box {
    background: #131925;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: clamp(1.15rem, 2.5vw, 1.8rem);
}
.setup-box h3 {
    color: #b7adff !important;
    margin-top: 1.2rem;
}
.setup-box h3:first-child {
    margin-top: 0;
}
.setup-box code {
    background: var(--bg-surface);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #b8c2ff;
}
.setup-box pre {
    background: #0e131c !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem;
    overflow-x: auto;
    font-size: 0.82rem;
}

.home-status-card {
    min-height: 172px;
    margin-bottom: 0;
}
.home-model-name {
    color: var(--text-hi);
    font-family: var(--font-display);
    font-size: 1.08rem;
    font-weight: 650;
    margin-bottom: 0.35rem;
}
.home-status-detail {
    color: var(--text-mid);
    font-size: 0.84rem;
    line-height: 1.55;
    margin: 0.75rem 0 0;
    overflow-wrap: anywhere;
}
.home-section {
    margin-top: 1.7rem;
    margin-bottom: 0.8rem;
}
.audio-model-note,
.audio-quality-note {
    color: var(--text-mid);
    font-size: 0.84rem;
    line-height: 1.6;
}
.audio-result {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.1rem;
}
.audio-result-emotion {
    color: var(--text-hi);
    font-family: var(--font-display);
    font-size: clamp(1.7rem, 3vw, 2.25rem);
    font-weight: 700;
    line-height: 1.1;
    text-transform: capitalize;
}
.audio-result-score {
    color: #c4bbff;
    font-size: 1.15rem;
    font-weight: 650;
    font-variant-numeric: tabular-nums;
    margin-top: 0.35rem;
}
.audio-score-note {
    color: var(--text-lo);
    font-size: 0.78rem;
    line-height: 1.5;
    margin: -0.3rem 0 1rem;
}
.audio-upload-card {
    padding: clamp(1.25rem, 3vw, 2rem);
}
.audio-meta-label {
    color: var(--text-lo);
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.audio-meta-value {
    color: var(--text-hi);
    font-size: 0.9rem;
    font-weight: 550;
    overflow-wrap: anywhere;
}

.sidebar-logo {
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: 700;
    color: #c4bbff;
    letter-spacing: -0.025em;
    padding: 0.9rem 0 0.45rem;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 5px;
}
[data-testid="stSidebar"] [role="radiogroup"] > label {
    padding: 0.65rem 0.75rem;
    border: 1px solid transparent;
    border-radius: 9px;
    color: var(--text-mid);
    transition: color 140ms ease, background-color 140ms ease, border-color 140ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.035);
    color: var(--text-hi);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: var(--accent-soft);
    border-color: rgba(154, 140, 255, 0.2);
    color: #d1cbff;
}
[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
    accent-color: var(--accent);
}

[data-testid="stButton"] > button,
[data-testid="stDownloadButton"] > button {
    min-height: 2.55rem;
    border: 1px solid #3a4558;
    border-radius: 9px;
    background: #1a2130;
    color: var(--text-hi);
    font-weight: 600;
    transition: border-color 140ms ease, background-color 140ms ease, transform 140ms ease;
}
[data-testid="stButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent);
    background: #232a3b;
    color: #fff;
    transform: translateY(-1px);
}
[data-testid="stButton"] > button:active,
[data-testid="stDownloadButton"] > button:active {
    transform: translateY(0);
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--bg-surface);
    border-color: #3a4558;
    color: var(--text-hi);
    border-radius: 8px;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover,
[data-testid="stTextInput"] input:hover,
[data-testid="stNumberInput"] input:hover,
[data-testid="stTextArea"] textarea:hover {
    border-color: #59657b;
}

input[type="range"] {
    accent-color: var(--accent);
}
[data-testid="stExpander"] {
    border-color: var(--border);
    border-radius: var(--radius-md);
    background: rgba(18, 23, 34, 0.58);
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
}
[data-testid="stAlert"] {
    border-radius: var(--radius-md);
}

:focus-visible {
    outline: 2px solid #b5aaff !important;
    outline-offset: 3px !important;
}

[data-testid="stHorizontalBlock"] {
    gap: clamp(0.75rem, 1.5vw, 1.25rem);
}

@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1.1rem 1rem 2rem;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
    }
    [data-testid="column"] {
        min-width: min(100%, 250px);
    }
    .hero-banner {
        margin-bottom: 1.2rem;
    }
    .prob-label {
        width: 84px;
        font-size: 0.78rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

MPL_STYLE = {
    "figure.facecolor": "#161b27", "axes.facecolor": "#1e2535",
    "axes.edgecolor": "#2a3450",   "axes.labelcolor": "#9ba6c2",
    "axes.grid": True,             "grid.color": "#2a3450",
    "grid.linewidth": 0.6,         "xtick.color": "#9ba6c2",
    "ytick.color": "#9ba6c2",      "text.color": "#f0f4ff",
    "font.family": "sans-serif",   "font.size": 10,
}
EMOTION_EMOJIS = {
    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}
ACCENT = "#7c6af7"
ACCENT2 = "#5ba3f5"
COND_LABELS = {"clean": "Clean", "snr_20db": "20 dB SNR",
               "snr_10db": "10 dB SNR", "snr_0db": "0 dB SNR"}
COND_ORDER = ["clean", "snr_20db", "snr_10db", "snr_0db"]


def model_ready() -> bool:
    return not missing_model_artifacts(ORIGINAL_CNN_LSTM)


def evaluation_ready() -> bool:
    try:
        _load_svm_noise_evaluation()
        return True
    except (
        FileNotFoundError, OSError, ValueError, TypeError, KeyError, IndexError,
        OverflowError, AttributeError, UnicodeError, pd.errors.ParserError,
    ):
        return False


def final_clean_comparison_ready() -> bool:
    try:
        _load_final_clean_comparison()
        return True
    except (
        FileNotFoundError, OSError, ValueError, TypeError, KeyError, IndexError,
        OverflowError, AttributeError, UnicodeError, pd.errors.ParserError,
    ):
        return False


@st.cache_data(show_spinner=False)
def dataset_inventory() -> tuple[int, int]:
    dataset = discover_ravdess()
    if dataset.empty:
        return 0, 0
    return int(len(dataset)), int(dataset["actor_id"].nunique())


ANALYTICS_CONDITIONS = {"clean", "snr_20db", "snr_10db", "snr_0db"}
ANALYTICS_METRICS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
)
COMPARISON_MODEL_CONTEXTS = {
    "original": {
        "label": ORIGINAL_CNN_LSTM,
        "manifest": "test_manifest.csv",
        "artifacts": ("emotionsense_cnn_lstm.keras", "label_mapping.json"),
        "features": "Per-recording normalized MFCC sequence",
    },
    "svm": {
        "label": SVM_BASELINE,
        "manifest": "test_manifest.csv",
        "artifacts": ("svm_model.pkl", "svm_scaler.pkl", "svm_label_mapping.json"),
        "features": "MFCC mean, standard deviation, minimum, and maximum; saved scaler",
    },
    "improved": {
        "label": "Improved CNN-LSTM",
        "manifest": "improved_test_manifest.csv",
        "artifacts": ("improved_cnn_lstm.keras", "improved_label_mapping.json"),
        "features": "MFCC, delta, and delta-delta sequence",
    },
}


def _read_actor_split() -> dict[str, list[int]]:
    split_path = MODEL_DIR / "split_actors.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing provenance artifact: {split_path.name}")
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        parsed = {
            name: [int(actor) for actor in split[name]]
            for name in ("train", "validation", "test")
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Invalid actor split file {split_path.name}: {exc}") from exc
    actor_sets = [set(parsed[name]) for name in ("train", "validation", "test")]
    if any(not actors for actors in actor_sets):
        raise ValueError(f"Actor split file {split_path.name} contains an empty split.")
    if any(len(actors) != len(parsed[name]) for actors, name in zip(
        actor_sets, ("train", "validation", "test")
    )):
        raise ValueError(f"Actor split file {split_path.name} contains duplicate actors.")
    if (actor_sets[0] & actor_sets[1]) or (actor_sets[0] & actor_sets[2]) or (actor_sets[1] & actor_sets[2]):
        raise ValueError(f"Actor split file {split_path.name} assigns actors to multiple splits.")
    return parsed


def _read_test_manifest(
    manifest_name: str,
    test_actors: set[int],
    mapping: dict[str, str],
    model_class_ids: Iterable[int],
) -> tuple[set[tuple[str, int]], int]:
    manifest_path = OUTPUT_DIR / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing provenance artifact: {manifest_path.name}")
    try:
        manifest = pd.read_csv(manifest_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read {manifest_path.name}: {exc}") from exc
    if manifest.empty or not {"path", "y_true"}.issubset(manifest.columns):
        raise ValueError(
            f"{manifest_path.name} must contain non-empty path and y_true columns."
        )

    validated_labels = validate_test_manifest_labels(
        manifest[["path", "y_true"]].itertuples(index=False, name=None),
        mapping,
        model_class_ids,
        manifest_path.name,
    )
    entries = set()
    manifest_rows = manifest[["path", "y_true"]].itertuples(index=False, name=None)
    for (raw_path, _), label in zip(manifest_rows, validated_labels):
        audio_path = Path(str(raw_path))
        parsed = parse_ravdess_filename(audio_path)
        if parsed is None or parsed["actor_id"] not in test_actors:
            raise ValueError(
                f"{manifest_path.name} contains a non-RAVDESS path or an actor "
                "outside the saved test split."
            )
        if not audio_path.is_file():
            raise ValueError(
                f"{manifest_path.name} references a missing recording: {audio_path.name}"
            )
        entries.add((str(audio_path.resolve()).casefold(), label))
    if len(entries) != len(manifest):
        raise ValueError(f"{manifest_path.name} contains duplicate path/label entries.")
    return entries, len(manifest)


def _validate_evaluation_context(
    result_path: Path, model_key: str, manifest_name: str
) -> tuple[set[tuple[str, int]], int, list[int], dict[str, str]]:
    context = COMPARISON_MODEL_CONTEXTS[model_key]
    required_paths = [MODEL_DIR / name for name in context["artifacts"]]
    required_paths += [MODEL_DIR / "split_actors.json", OUTPUT_DIR / manifest_name]
    missing = [
        f"{path.parent.name}/{path.name}"
        for path in required_paths if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts needed to verify evaluation provenance: "
            + ", ".join(missing)
        )
    split = _read_actor_split()
    mapping_path = MODEL_DIR / context["artifacts"][-1]
    if model_key == "svm":
        mapping_path = MODEL_DIR / "svm_label_mapping.json"
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read label mapping {mapping_path.name}: {exc}") from exc
    manifest_entries, manifest_count = _read_test_manifest(
        manifest_name,
        set(split["test"]),
        mapping,
        range(len(mapping)) if isinstance(mapping, dict) else (),
    )

    stale = [
        f"{path.parent.name}/{path.name}" for path in required_paths
        if result_path.stat().st_mtime_ns < path.stat().st_mtime_ns
    ]
    if stale:
        raise ValueError(
            f"{result_path.name} predates current provenance artifact(s): "
            + ", ".join(stale)
            + ". Regenerate evaluation results after restoring the intended artifacts."
        )
    return manifest_entries, manifest_count, split["test"], mapping


def _read_metrics_csv(path: Path, *, comparison: bool = False) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    required = {"condition", "n_test_files", "run_id", *ANALYTICS_METRICS}
    if comparison:
        required |= {"model", "model_key"}
    if frame.empty or not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(
            f"{path.name} is empty or missing required column(s): "
            + (", ".join(missing) or "no rows")
        )
    if frame["condition"].isna().any() or not set(frame["condition"]).issubset(ANALYTICS_CONDITIONS):
        raise ValueError(f"{path.name} contains an unknown or empty noise condition.")
    if frame["run_id"].isna().any() or frame["run_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{path.name} contains an empty run_id.")
    if comparison and (
        frame["model"].isna().any()
        or frame["model_key"].isna().any()
        or frame["model"].astype(str).str.strip().eq("").any()
        or frame["model_key"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"{path.name} contains an empty model name or key.")
    numeric_columns = ["n_test_files", *ANALYTICS_METRICS]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name} contains non-numeric or non-finite {column} values.")
    counts = frame["n_test_files"].to_numpy(dtype=float)
    if np.any(counts <= 0) or np.any(counts != np.floor(counts)):
        raise ValueError(f"{path.name} contains invalid test-file counts.")
    for metric in ANALYTICS_METRICS:
        if ((frame[metric] < 0) | (frame[metric] > 1)).any():
            raise ValueError(f"{path.name} contains {metric} values outside [0, 1].")
    if comparison and frame.duplicated(["model_key", "condition"]).any():
        raise ValueError(f"{path.name} contains duplicate model/condition rows.")
    if not comparison and frame["condition"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate noise-condition rows.")
    return frame


def _read_evaluation_predictions(path: Path) -> list[dict]:
    try:
        frame = pd.read_csv(
            path,
            dtype={
                "noise_base_seed": "string",
                "noise_seed": "string",
                "recording_sha256": "string",
            },
        )
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    required = {
        "run_id", "model_key", "condition", "path", "y_true", "true_label",
        "y_pred", "predicted_label", "status", "error",
        "noise_base_seed", "noise_seed", "recording_sha256",
    }
    if frame.empty or not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(
            f"{path.name} is empty or missing required column(s): "
            + (", ".join(missing) or "no rows")
        )
    for column in ("run_id", "model_key", "condition", "path", "true_label", "status"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{path.name} contains an empty {column} value.")
    for column in ("y_true", "y_pred"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    true_labels = frame["y_true"].to_numpy(dtype=float)
    if not np.isfinite(true_labels).all() or np.any(true_labels != np.floor(true_labels)):
        raise ValueError(f"{path.name} contains invalid y_true class IDs.")
    evaluated = frame["status"].eq("evaluated")
    predictions = frame.loc[evaluated, "y_pred"].to_numpy(dtype=float)
    failures = frame.loc[~evaluated, "error"]
    if (
        not np.isfinite(predictions).all()
        or np.any(predictions != np.floor(predictions))
        or frame.loc[~evaluated, "y_pred"].notna().any()
        or frame.loc[evaluated, "predicted_label"].isna().any()
        or frame.loc[evaluated, "predicted_label"].astype(str).str.strip().eq("").any()
        or frame.loc[~evaluated, "predicted_label"].notna().any()
        or failures.isna().any()
        or failures.astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"{path.name} contains invalid or misaligned predictions.")
    return frame.to_dict(orient="records")


def _validate_run_provenance(
    metrics: pd.DataFrame, records: list[dict], evaluator: str
) -> dict:
    run_ids = metrics["run_id"].astype(str).tolist()
    evidence_ids = [str(record["run_id"]) for record in records]
    unique_ids = set(run_ids)
    if len(unique_ids) != 1 or "" in unique_ids:
        raise ValueError("Aggregate metrics must belong to exactly one evaluation run.")
    run_id = next(iter(unique_ids))
    metadata_path = run_provenance_path(OUTPUT_DIR, evaluator, run_id)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read run provenance file {metadata_path.name}: {exc}"
        ) from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("status") != "completed"
        or metadata.get("evaluator") != evaluator
    ):
        raise ValueError(
            f"Run provenance {metadata_path.name} does not describe a completed run."
        )
    if not isinstance(metadata.get("inputs"), dict):
        raise ValueError(f"Run provenance {metadata_path.name} has no input fingerprint map.")
    validate_run_binding(run_ids, evidence_ids, metadata.get("run_id"))
    for record in records:
        fingerprint = record["recording_sha256"]
        path = Path(str(record["path"]))
        key = input_key(path, PROJECT_ROOT)
        input_entry = metadata.get("inputs", {}).get(key)
        if pd.isna(fingerprint) or not str(fingerprint).strip():
            if input_entry is None or input_entry.get("status") != "missing":
                raise ValueError(
                    "Per-record evidence is missing a recording fingerprint that "
                    "should be present in run metadata."
                )
            continue
        if (
            len(str(fingerprint)) != 64
            or any(character not in "0123456789abcdef" for character in str(fingerprint).lower())
            or input_entry is None
            or len(str(input_entry.get("sha256", ""))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(input_entry.get("sha256", "")).lower()
            )
            or input_entry.get("sha256") != str(fingerprint)
        ):
            raise ValueError(
                "Per-record recording fingerprint does not match run provenance."
            )
    return metadata


def _sample_records_match_manifest(
    records: list[dict],
    *,
    group_columns: tuple[str, ...],
    group: tuple[str, ...],
    manifest_entries: set[tuple[str, int]],
) -> bool:
    actual = set()
    for record in records:
        if tuple(str(record[column]) for column in group_columns) != group:
            continue
        try:
            label = int(record["y_true"])
            if float(record["y_true"]) != label:
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        actual.add((Path(str(record["path"])).resolve().as_posix().casefold(), label))
    expected = {
        (Path(path).resolve().as_posix().casefold(), int(label))
        for path, label in manifest_entries
    }
    return actual == expected and len(actual) == sum(
        1 for record in records
        if tuple(str(record[column]) for column in group_columns) == group
    )


def _validate_sample_metric_counts(
    metrics: pd.DataFrame,
    records: list[dict],
    group_columns: tuple[str, ...],
) -> None:
    evaluated_counts: dict[tuple[str, ...], int] = {}
    for record in records:
        if record["status"] == "evaluated":
            group = tuple(str(record[column]) for column in group_columns)
            evaluated_counts[group] = evaluated_counts.get(group, 0) + 1
    for row in metrics.to_dict(orient="records"):
        group = tuple(str(row[column]) for column in group_columns)
        if int(row["n_test_files"]) != evaluated_counts.get(group, 0):
            raise ValueError(
                f"Metric count for {group} does not match evaluated prediction identities."
            )


def _validate_prediction_label_mapping(
    records: list[dict], model_key: str, mapping: dict[str, str]
) -> None:
    for record in records:
        if str(record["model_key"]) != model_key:
            continue
        true_id = int(record["y_true"])
        if mapping.get(str(true_id)) != str(record["true_label"]):
            raise ValueError(
                f"Prediction details for {model_key} contain a true label that "
                "does not match the model's label mapping."
            )
        if record["status"] == "evaluated":
            predicted_id = int(record["y_pred"])
            if (
                str(predicted_id) not in mapping
                or mapping[str(predicted_id)] != str(record["predicted_label"])
            ):
                raise ValueError(
                    f"Prediction details for {model_key} contain an invalid predicted label."
                )


def _validate_noise_seed_evidence(records: list[dict]) -> None:
    snr_by_condition = {
        "snr_20db": 20,
        "snr_10db": 10,
        "snr_0db": 0,
    }
    for record in records:
        condition = str(record["condition"])
        base_value = record["noise_base_seed"]
        seed_value = record["noise_seed"]
        if condition == "clean":
            if pd.notna(base_value) or pd.notna(seed_value):
                raise ValueError("Clean-condition evidence must not contain a noise seed.")
            continue
        if condition not in snr_by_condition:
            raise ValueError(f"Unknown noise condition in prediction evidence: {condition}.")
        if pd.isna(base_value) or pd.isna(seed_value):
            if record["status"] in {"evaluated", "inference_error"}:
                raise ValueError(
                    "A processed noisy recording is missing its deterministic noise seed."
                )
            continue
        try:
            base_seed = int(str(base_value))
            actual_seed = int(str(seed_value))
            if str(base_seed) != str(base_value) or str(actual_seed) != str(seed_value):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Prediction evidence contains an invalid noise seed.") from exc
        expected_seed = derive_noise_seed(
            str(record["path"]), condition, base_seed
        )
        if actual_seed != expected_seed:
            raise ValueError(
                "Prediction evidence noise seed does not match its recording, "
                "condition, and base seed."
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_svm_noise_evaluation() -> tuple[pd.DataFrame, list[int], int, dict]:
    metrics_path = OUTPUT_DIR / "all_models_comparison.csv"
    predictions_path = OUTPUT_DIR / "all_models_predictions.csv"
    metrics = _read_metrics_csv(metrics_path, comparison=True)
    records = _read_evaluation_predictions(predictions_path)
    _validate_noise_seed_evidence(records)
    provenance = _validate_run_provenance(
        metrics, records, "all_models_evaluation"
    )
    expected_conditions = set(ANALYTICS_CONDITIONS)
    if provenance["run_id"] != "695324b2e90a4af5b72034c5b95f5ebb":
        raise ValueError("The SVM noise evidence is not from the verified evaluation run.")
    if (
        set(metrics["model_key"]) != {"svm"}
        or not metrics["model"].eq(SVM_BASELINE).all()
        or set(metrics["condition"]) != expected_conditions
        or set(zip(metrics["model_key"], metrics["condition"]))
        != {("svm", condition) for condition in expected_conditions}
    ):
        raise ValueError("Noise-evaluation metrics must contain only the four SVM conditions.")
    if any(str(record["model_key"]) != "svm" for record in records):
        raise ValueError("Noise-evaluation prediction evidence contains a non-SVM model.")

    manifest_entries, manifest_count, test_actors, mapping = _validate_evaluation_context(
        metrics_path, "svm", "test_manifest.csv"
    )
    if manifest_count != 300:
        raise ValueError("The verified SVM noise evaluation must use 300 test recordings.")
    _validate_prediction_label_mapping(records, "svm", mapping)
    if not evaluation_samples_match(
        records,
        ["model_key", "condition"],
        expected_groups=[("svm", condition) for condition in sorted(expected_conditions)],
    ):
        raise ValueError("SVM noise evidence does not use identical recordings per condition.")
    if any(
        not _sample_records_match_manifest(
            records,
            group_columns=("model_key", "condition"),
            group=("svm", condition),
            manifest_entries=manifest_entries,
        )
        for condition in expected_conditions
    ):
        raise ValueError("SVM noise evidence does not match the held-out test manifest.")
    _validate_sample_metric_counts(metrics, records, ("model_key", "condition"))
    if (metrics["n_test_files"] != manifest_count).any():
        raise ValueError("SVM noise metrics do not cover all 300 test recordings.")
    expected_accuracy = {
        "clean": 0.47,
        "snr_20db": 0.23666666666666666,
        "snr_10db": 0.14,
        "snr_0db": 0.13333333333333333,
    }
    observed_accuracy = metrics.set_index("condition")["accuracy"].to_dict()
    if any(
        not np.isclose(observed_accuracy[condition], expected, rtol=0, atol=1e-12)
        for condition, expected in expected_accuracy.items()
    ):
        raise ValueError("SVM noise metrics do not match the verified accuracy results.")
    if metrics_path.stat().st_mtime_ns < predictions_path.stat().st_mtime_ns:
        raise ValueError("SVM metrics predate their per-record prediction evidence.")
    return metrics, test_actors, manifest_count, provenance


def _load_final_clean_comparison() -> dict:
    experiment_dir = PROJECT_ROOT / "experiments" / "final_model_comparison"
    result_path = experiment_dir / "final_comparison.json"
    predictions_path = experiment_dir / "final_predictions.csv"
    manifest_path = OUTPUT_DIR / "test_manifest.csv"
    split_path = MODEL_DIR / "split_actors.json"
    improved_dir = PROJECT_ROOT / "experiments" / "improved_cnn_lstm_run_01"
    artifact_paths = {
        "svm_model": MODEL_DIR / "svm_model.pkl",
        "svm_scaler": MODEL_DIR / "svm_scaler.pkl",
        "svm_label_mapping": MODEL_DIR / "svm_label_mapping.json",
        "cnn_lstm_model": improved_dir / "improved_cnn_lstm.keras",
        "cnn_lstm_label_mapping": improved_dir / "label_mapping.json",
        "test_manifest": manifest_path,
        "actor_split": split_path,
    }
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {result_path.name}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{result_path.name} must contain a JSON object.")
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Final comparison has no provenance object.")

    try:
        split = _read_actor_split()
        recorded_split = provenance["actor_split"]
        recorded_split = {
            name: [int(actor) for actor in recorded_split[name]]
            for name in ("train", "validation", "test")
        }
        fingerprints = provenance["artifact_sha256"]
        class_order = provenance["class_order"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Final comparison provenance is incomplete or malformed.") from exc
    if recorded_split != split:
        raise ValueError("Final comparison actor split differs from the saved split.")
    if (
        not isinstance(class_order, list)
        or len(class_order) != 8
        or any(not isinstance(label, str) for label in class_order)
        or len(set(class_order)) != 8
        or class_order != [
            "angry", "calm", "disgust", "fearful",
            "happy", "neutral", "sad", "surprised",
        ]
    ):
        raise ValueError("Final comparison must declare its eight-class order.")
    if (
        provenance.get("test_record_count") != 300
        or provenance.get("evaluation") != "clean audio only; no augmentation, tuning, or retraining"
        or not isinstance(provenance.get("run_timestamp_utc"), str)
        or not provenance["run_timestamp_utc"].strip()
    ):
        raise ValueError("Final comparison does not describe the required clean 300-record run.")
    if not isinstance(fingerprints, dict):
        raise ValueError("Final comparison has no artifact fingerprints.")
    for key, path in artifact_paths.items():
        expected_hash = fingerprints.get(key)
        if (
            not path.is_file()
            or not isinstance(expected_hash, str)
            or _sha256_file(path) != expected_hash
        ):
            raise ValueError(f"Final comparison fingerprint mismatch for {path.name}.")
    if provenance.get("test_manifest_sha256") != fingerprints.get("test_manifest"):
        raise ValueError("Final comparison manifest fingerprints are inconsistent.")

    try:
        mappings = {
            "svm_baseline": json.loads(
                artifact_paths["svm_label_mapping"].read_text(encoding="utf-8")
            ),
            "improved_cnn_lstm": json.loads(
                artifact_paths["cnn_lstm_label_mapping"].read_text(encoding="utf-8")
            ),
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read a final-comparison label mapping.") from exc
    expected_mapping = {str(index): label for index, label in enumerate(class_order)}
    if any(mapping != expected_mapping for mapping in mappings.values()):
        raise ValueError("Final comparison model label mappings do not match its class order.")

    manifest_entries, manifest_count = _read_test_manifest(
        "test_manifest.csv",
        set(split["test"]),
        mappings["svm_baseline"],
        range(len(class_order)),
    )
    if manifest_count != 300 or len(split["test"]) != 5:
        raise ValueError("Final comparison manifest is not the expected held-out test set.")
    expected_actor_counts = {str(actor): 60 for actor in split["test"]}
    if provenance.get("test_actor_counts") != expected_actor_counts:
        raise ValueError("Final comparison actor-count provenance is inconsistent.")
    actual_actor_counts: dict[str, int] = {}
    for path, _ in manifest_entries:
        parsed = parse_ravdess_filename(Path(path))
        if parsed is None:
            raise ValueError("Final comparison manifest contains an invalid RAVDESS filename.")
        actor = str(parsed["actor_id"])
        actual_actor_counts[actor] = actual_actor_counts.get(actor, 0) + 1
    if actual_actor_counts != expected_actor_counts:
        raise ValueError("Final comparison manifest actor counts do not match the saved split.")

    try:
        predictions = pd.read_csv(predictions_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read {predictions_path.name}: {exc}") from exc
    required_columns = {
        "path", "actor_id", "y_true", "true_emotion", "svm_y_pred", "svm_emotion",
        "cnn_lstm_y_pred", "cnn_lstm_emotion",
    }
    if (
        not required_columns.issubset(predictions.columns)
        or len(predictions) != manifest_count
        or predictions["path"].isna().any()
        or predictions["path"].duplicated().any()
    ):
        raise ValueError("Final comparison prediction evidence is incomplete or duplicated.")
    prediction_identities = set()
    for row in predictions.to_dict(orient="records"):
        parsed = parse_ravdess_filename(Path(str(row["path"])))
        if parsed is None or int(row["actor_id"]) != parsed["actor_id"]:
            raise ValueError("Final predictions contain an invalid actor identity.")
        try:
            true_id = int(row["y_true"])
            if float(row["y_true"]) != true_id:
                raise ValueError
            predictions_by_model = {
                "svm_baseline": int(row["svm_y_pred"]),
                "improved_cnn_lstm": int(row["cnn_lstm_y_pred"]),
            }
            if any(
                float(row[column]) != prediction
                for column, prediction in (
                    ("svm_y_pred", predictions_by_model["svm_baseline"]),
                    ("cnn_lstm_y_pred", predictions_by_model["improved_cnn_lstm"]),
                )
            ):
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Final predictions contain a non-integer class ID.") from exc
        if (
            str(parsed["actor_id"]) not in expected_actor_counts
            or not 0 <= true_id < len(class_order)
            or str(row["true_emotion"]) != class_order[true_id]
        ):
            raise ValueError("Final predictions contain an invalid ground-truth label or actor.")
        identity = (str(Path(str(row["path"])).resolve()).casefold(), true_id)
        if identity not in manifest_entries:
            raise ValueError("Final predictions do not match the held-out manifest.")
        prediction_identities.add(identity)
        for model_key, predicted_id in predictions_by_model.items():
            if (
                not 0 <= predicted_id < len(class_order)
                or str(row[f"{'svm' if model_key == 'svm_baseline' else 'cnn_lstm'}_emotion"])
                != class_order[predicted_id]
            ):
                raise ValueError("Final predictions contain an invalid predicted label.")
    if prediction_identities != manifest_entries or len(prediction_identities) != manifest_count:
        raise ValueError("Final predictions do not cover the exact held-out manifest.")

    model_summaries = result.get("models")
    expected_model_keys = {"svm_baseline", "improved_cnn_lstm"}
    if not isinstance(model_summaries, dict) or set(model_summaries) != expected_model_keys:
        raise ValueError("Final comparison is missing one or both model results.")
    expected_metrics = {
        "svm_baseline": {"accuracy": 0.47, "macro_f1": 0.4556359440137682},
        "improved_cnn_lstm": {
            "accuracy": 0.18333333333333332,
            "macro_f1": 0.16630446685639383,
        },
    }
    for model_key in expected_model_keys:
        summary = model_summaries[model_key]
        if (
            not isinstance(summary, dict)
            or summary.get("class_order") != class_order
            or summary.get("total_count") != manifest_count
            or not isinstance(summary.get("correct_count"), int)
        ):
            raise ValueError(f"Final comparison summary is inconsistent for {model_key}.")
        for metric, expected in expected_metrics[model_key].items():
            if (
                not isinstance(summary.get(metric), (int, float))
                or not np.isclose(summary[metric], expected, rtol=0, atol=1e-12)
            ):
                raise ValueError(f"Final comparison {metric} does not match the verified result.")
        report = summary.get("classification_report")
        matrix = summary.get("confusion_matrix")
        if (
            not isinstance(report, dict)
            or any(not isinstance(report.get(label), dict) for label in class_order)
            or not isinstance(matrix, list)
            or len(matrix) != len(class_order)
            or any(not isinstance(row, list) or len(row) != len(class_order) for row in matrix)
        ):
            raise ValueError(f"Final comparison class evidence is inconsistent for {model_key}.")
        report_macro = report.get("macro avg")
        if (
            not isinstance(report_macro, dict)
            or not np.isclose(
                float(report_macro.get("f1-score", float("nan"))),
                summary["macro_f1"], rtol=0, atol=1e-12,
            )
            or not np.isclose(
                float(report.get("accuracy", float("nan"))),
                summary["accuracy"], rtol=0, atol=1e-12,
            )
        ):
            raise ValueError(f"Final comparison report metrics conflict for {model_key}.")
    return result


def open_audio_intelligence() -> None:
    st.session_state["page_navigation"] = "🎤  Analyze Audio"


def predict_audio_bytes(audio_bytes: bytes, model_name: str = ORIGINAL_CNN_LSTM) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_path = Path(temp_file.name)
    try:
        temp_path.write_bytes(audio_bytes)
        return predict_file(temp_path, model_name)
    finally:
        temp_path.unlink(missing_ok=True)


def waveform_fig(audio: np.ndarray, sr: int) -> plt.Figure:
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(8, 2.4))
        times = np.arange(len(audio)) / sr
        ax.plot(times, audio, linewidth=0.55, color=ACCENT, alpha=0.9)
        ax.fill_between(times, audio, alpha=0.15, color=ACCENT2)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_xlim(0, max(times[-1], 0.01))
        ax.set_ylim(-1.05, 1.05)
        fig.tight_layout(pad=0.4)
    return fig


def mel_spectrogram_fig(audio: np.ndarray, sr: int) -> plt.Figure:
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(8, 2.8))
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=160,
                                       x_axis="time", y_axis="mel", ax=ax,
                                       fmax=8000, cmap="magma")
        fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        fig.tight_layout(pad=0.4)
    return fig


def probability_bars_html(class_probs: list[dict]) -> str:
    rows = ""
    for item in sorted(class_probs, key=lambda x: x["score"], reverse=True):
        pct = item["score"] * 100
        emoji = EMOTION_EMOJIS.get(item["emotion"], "")
        rows += (
            f'<div class="prob-row">'
            f'<span class="prob-label">{emoji} {item["emotion"]}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%"></div></div>'
            f'<span class="prob-pct">{pct:.1f}%</span></div>'
        )
    return rows


def segment_timeline_html(audio: np.ndarray, sr: int, seg_sec: float = 2.5,
                          model_name: str = ORIGINAL_CNN_LSTM) -> str:
    import soundfile as sf
    seg_len = int(seg_sec * sr)
    segments = []
    for start in range(0, len(audio) - seg_len + 1, seg_len):
        segments.append((start / sr, (start + seg_len) / sr, audio[start:start + seg_len]))
    if not segments:
        return ""
    rows_html = ""
    for t0, t1, seg in segments:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, seg, sr)
            res = predict_file(tmp_path, model_name)
            em = res["predicted_emotion"]
            score = res["top_score"] * 100
            emoji = EMOTION_EMOJIS.get(em, "")
            rows_html += (
                f'<div class="seg-row">'
                f'<span class="seg-time">{t0:.1f}–{t1:.1f}s</span>'
                f'<span class="seg-label">{emoji} {em.capitalize()}</span>'
                f'<span class="seg-conf">{score:.0f}% score</span></div>'
            )
        except Exception as exc:
            st.warning(f"Could not predict segment at {t0:.1f}s: {exc}")
            rows_html += (
                f'<div class="seg-row">'
                f'<span class="seg-time">{t0:.1f}–{t1:.1f}s</span>'
                f'<span class="seg-label">Unavailable</span></div>'
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return rows_html


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎙️ EmotionSense</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Studio Home", "🎤  Analyze Audio", "🔬  Noise Demo",
         "📊  Noise Robustness", "📈  Model Comparison",
         "ℹ️  Setup Guide"],
        label_visibility="collapsed",
        key="page_navigation",
    )
    st.markdown("---")
    _models_avail = available_prediction_models()
    st.markdown(
        f'<p style="font-size:0.75rem;color:#4a5568;line-height:1.7;">'
        f"EmotionSense Studio<br>"
        f"Final model: {SVM_BASELINE}<br>"
        f"SVM noise results: {'available' if evaluation_ready() else 'unavailable'}<br>"
        f"Clean model comparison: {'available' if final_clean_comparison_ready() else 'unavailable'}<br>"
        f"RAVDESS · Local CPU<br>Academic mini-project</p>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

_models_avail = available_prediction_models()
_ready = bool(_models_avail)
_n_models = len(_models_avail)
_cls = "status-ready" if _ready else "status-missing"
_dot_extra = "" if _ready else " status-dot-pulse"
_msg = (f"{_n_models} prediction model(s) ready — {', '.join(_models_avail)}" if _ready
        else "No complete prediction model artifacts found")

st.markdown(
    f"""<div class="hero-banner">
        <div class="hero-title">EmotionSense Studio</div>
        <p class="hero-sub">
            RAVDESS speech emotion recognition · Final selected model: SVM Baseline · CNN-LSTM variants are experimental
        </p>
        <span class="status-pill {_cls}">
            <span class="status-dot{_dot_extra}"></span>{_msg}
        </span>
    </div>""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — ANALYZE AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

if "🏠" in page:
    st.markdown("## Your speech emotion workspace")
    st.markdown(
        "Explore audio emotion predictions and controlled noise experiments. "
        "Model and evaluation status below reflects files currently available in this project."
    )

    st.markdown('<h3 class="home-section">Project inventory</h3>', unsafe_allow_html=True)
    try:
        with st.spinner("Scanning the project data for valid RAVDESS speech recordings…"):
            dataset_files, dataset_actors = dataset_inventory()
    except Exception as exc:
        st.error(f"Could not inspect the RAVDESS dataset inventory: {exc}")
    else:
        if dataset_files:
            inventory_cols = st.columns(2)
            inventory_cols[0].metric("Valid RAVDESS speech files", f"{dataset_files:,}")
            inventory_cols[1].metric("Distinct actors found", f"{dataset_actors:,}")
            st.caption(
                "Live count from recursive filename-based discovery under data/. "
                "Only recognized RAVDESS audio-only speech files are counted."
            )
        else:
            st.warning(
                "No valid RAVDESS speech recordings were discovered under data/. "
                "Dataset counts are unavailable."
            )

    st.markdown('<h3 class="home-section">Prediction models</h3>', unsafe_allow_html=True)
    original_missing = missing_model_artifacts(ORIGINAL_CNN_LSTM)
    svm_missing = missing_model_artifacts(SVM_BASELINE)
    improved_paths = (
        PROJECT_ROOT / "experiments" / "improved_cnn_lstm_run_01" / "improved_cnn_lstm.keras",
        PROJECT_ROOT / "experiments" / "improved_cnn_lstm_run_01" / "label_mapping.json",
    )
    improved_missing = [path.name for path in improved_paths if not path.is_file()]
    model_cards = [
        (
            "SVM Baseline",
            "Available" if not svm_missing else "Unavailable",
            "Ready for uploaded-audio prediction."
            if not svm_missing else f"Missing required artifact(s): {', '.join(svm_missing)}",
            not svm_missing,
        ),
        (
            "Original CNN-LSTM",
            "Available" if not original_missing else "Unavailable",
            "Ready for uploaded-audio prediction."
            if not original_missing else f"Missing required artifact(s): {', '.join(original_missing)}",
            not original_missing,
        ),
        (
            "Improved CNN-LSTM",
            "Experimental artifacts available" if not improved_missing else "Unavailable",
            "Weights are present in experiments/improved_cnn_lstm_run_01/, but this model is not integrated into uploaded-audio inference."
            if not improved_missing else f"Missing experiment artifact(s): {', '.join(improved_missing)}",
            not improved_missing,
        ),
    ]
    model_cols = st.columns(3)
    for column, (model_name, status, detail, ready) in zip(model_cols, model_cards):
        state_class = "status-ready" if ready else "status-missing"
        with column:
            st.markdown(
                f"""<div class="es-card home-status-card">
                    <div class="es-card-title">Prediction model</div>
                    <div class="home-model-name">{model_name}</div>
                    <span class="status-pill {state_class}">{status}</span>
                    <p class="home-status-detail">{detail}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<h3 class="home-section">Evaluation artifacts</h3>', unsafe_allow_html=True)
    eval_cols = st.columns(2)
    with eval_cols[0]:
        if evaluation_ready():
            st.success(
                "Verified SVM-only noise-evaluation results are available "
                "(run 695324b2e90a4af5b72034c5b95f5ebb)."
            )
        else:
            st.info(
                "Verified SVM noise-evaluation evidence is unavailable or failed validation."
            )
    with eval_cols[1]:
        if final_clean_comparison_ready():
            st.success(
                "The separate clean SVM-versus-improved-CNN-LSTM comparison is available."
            )
        else:
            st.info(
                "The final clean model comparison is unavailable or failed validation."
            )

    st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)
    action_col, detail_col = st.columns([1, 2])
    with action_col:
        st.button(
            "Open Audio Intelligence",
            type="primary",
            use_container_width=True,
            on_click=open_audio_intelligence,
        )
    with detail_col:
        st.caption(
            "Upload speech to inspect its waveform and spectrogram and run a prediction "
            "with an available model. Prediction scores are not calibration-validated."
        )

elif "🎤" in page:
    st.markdown("## Audio Intelligence")
    st.markdown(
        "Upload a speech recording to inspect its signal and run emotion inference. "
        "Predictions are model outputs, not a diagnosis or a measure of a speaker's true emotional state."
    )

    if not _models_avail:
        missing_cnn = missing_model_artifacts(ORIGINAL_CNN_LSTM)
        missing_svm = missing_model_artifacts(SVM_BASELINE)
        st.markdown(
            f"""<div class="setup-box">
            <h3>No prediction model is currently available</h3>
            <p>Original CNN-LSTM is missing: {', '.join(missing_cnn) or 'none'}<br>
            SVM Baseline is missing: {', '.join(missing_svm) or 'none'}</p>
            <p>Prediction is disabled until the listed artifacts are available.
            No model weights or results are downloaded or generated by this page.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        model_col, model_note_col = st.columns([1, 2])
        with model_col:
            default_model_index = (
                _models_avail.index(SVM_BASELINE) if SVM_BASELINE in _models_avail else 0
            )
            selected_model = st.selectbox(
                "Prediction model",
                options=_models_avail,
                index=default_model_index,
            )
            st.caption("SVM Baseline is the final selected model; CNN-LSTM variants are experimental.")
        with model_note_col:
            if selected_model == ORIGINAL_CNN_LSTM:
                st.markdown(
                    '<p class="audio-model-note"><strong>Feature pipeline:</strong> '
                    "per-recording normalized MFCC sequence. The model accepts at most "
                    f"{CNN_MAX_AUDIO_SECONDS:.1f} seconds; longer recordings are truncated "
                    "for its prediction.</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="audio-model-note"><strong>Feature pipeline:</strong> '
                    "MFCC mean, standard deviation, minimum, and maximum over the full "
                    "recording, followed by the saved scaler. This differs from the "
                    "CNN-LSTM sequence representation.</p>",
                    unsafe_allow_html=True,
                )
        with st.expander("Model methods and limitations"):
            st.markdown(
                "**SVM Baseline:** summarizes MFCC features with mean, standard deviation, "
                "minimum, and maximum across the full recording, then applies its saved scaler. "
                "Its probability estimates are not calibration-validated."
            )
            st.markdown(
                f"**Original CNN-LSTM:** uses a per-recording normalized MFCC sequence and "
                f"accepts up to {CNN_MAX_AUDIO_SECONDS:.1f} seconds (approximately two seconds) "
                "of audio. Longer recordings are truncated for its prediction; its softmax "
                "scores are uncalibrated."
            )
            cnn_missing = missing_model_artifacts(ORIGINAL_CNN_LSTM)
            if cnn_missing:
                st.info(
                    "Original CNN-LSTM prediction is currently unavailable. Missing artifact(s): "
                    + ", ".join(cnn_missing)
                )

        st.markdown("### Select a recording")
        uploaded = st.file_uploader(
            "Choose a speech recording",
            type=["wav", "mp3", "flac", "ogg"],
            help="Supported formats: WAV, MP3, FLAC, and OGG. Upload speech audio for best relevance.",
        )

        if not uploaded:
            st.markdown(
                """<div class="es-card audio-upload-card" style="text-align:center;">
                    <div style="font-size:2rem;margin-bottom:0.55rem;color:#c4bbff;">Audio analysis</div>
                    <div style="font-size:0.95rem;font-weight:550;color:#f3f4f8;">
                        Choose a recording to begin</div>
                    <div style="font-size:0.82rem;margin-top:0.45rem;color:#b0b8c8;">
                        WAV · MP3 · FLAC · OGG
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            suffix = Path(uploaded.name).suffix or ".wav"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    temp_path = Path(tmp.name)
                    tmp.write(uploaded.getvalue())

                try:
                    audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
                    if audio.size == 0:
                        raise ValueError("The decoded recording contains no audio samples.")
                except Exception as exc:
                    st.error(
                        "This audio could not be decoded. Check that the file is a valid, "
                        "non-empty WAV, MP3, FLAC, or OGG recording, then try another file."
                    )
                    with st.expander("Technical details"):
                        st.code(str(exc))
                else:
                    st.audio(uploaded)
                    st.markdown("### Recording")
                    meta_cols = st.columns(4)
                    meta_cols[0].metric("Duration", f"{len(audio) / sr:.2f} s")
                    meta_cols[1].metric("Analysis sample rate", f"{sr:,} Hz")
                    meta_cols[2].metric("Channels", "Mono")
                    meta_cols[3].metric("File size", f"{uploaded.size / (1024 * 1024):.2f} MB")
                    st.caption(f"File: {uploaded.name}")

                    tab_wave, tab_mel = st.tabs(["Waveform", "Mel spectrogram"])
                    with tab_wave:
                        st.caption(
                            "Amplitude over time for the mono audio resampled to the model's input rate."
                        )
                        fig_w = waveform_fig(audio, sr)
                        st.pyplot(fig_w, use_container_width=True)
                        plt.close(fig_w)
                    with tab_mel:
                        st.caption(
                            "Mel-frequency energy over time; this visualization is for inspection "
                            "and is not the SVM feature chart."
                        )
                        try:
                            fig_m = mel_spectrogram_fig(audio, sr)
                            st.pyplot(fig_m, use_container_width=True)
                            plt.close(fig_m)
                        except Exception as exc:
                            st.warning(
                                "The recording was decoded, but its Mel spectrogram could not be rendered."
                            )
                            with st.expander("Spectrogram details"):
                                st.code(str(exc))

                    st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)
                    try:
                        with st.spinner(f"Analyzing with {selected_model}…"):
                            result = predict_file(temp_path, selected_model)
                    except FileNotFoundError as exc:
                        st.error(
                            f"{selected_model} is no longer available because a required model "
                            "artifact is missing. Refresh the page after restoring the artifact."
                        )
                        with st.expander("Missing artifact details"):
                            st.code(str(exc))
                    except Exception as exc:
                        st.error(
                            f"{selected_model} could not analyze this recording. Check that the "
                            "model artifacts are valid and try another supported speech file."
                        )
                        with st.expander("Inference details"):
                            st.code(str(exc))
                    else:
                        emotion = result["predicted_emotion"]
                        top_score = result["top_score"]
                        emoji = EMOTION_EMOJIS.get(emotion, "")
                        aq = result.get("audio_quality", {})

                        col_pred, col_quality = st.columns([3, 2])
                        with col_pred:
                            st.markdown(
                                f"""<div class="es-card">
                                    <div class="es-card-title">Prediction result</div>
                                    <div class="audio-result">
                                        <div style="font-size:2.7rem;" aria-hidden="true">{emoji}</div>
                                        <div>
                                            <div style="font-size:0.8rem;color:#b0b8c8;">{result["model"]}</div>
                                            <div class="audio-result-emotion">{emotion}</div>
                                            <div class="audio-result-score">{top_score * 100:.1f}% score</div>
                                        </div>
                                    </div>
                                    <div class="audio-score-note">{result["score_description"]}. Scores are not calibration-validated confidence probabilities.</div>
                                    <div class="es-card-title">Scores by emotion</div>
                                    {probability_bars_html(result["class_scores"])}
                                </div>""",
                                unsafe_allow_html=True,
                            )

                        with col_quality:
                            st.markdown(
                                '<div class="es-card"><div class="es-card-title">Signal checks</div>',
                                unsafe_allow_html=True,
                            )
                            quality_cols = st.columns(2)
                            if "peak_amplitude" in aq:
                                quality_cols[0].metric("Peak amplitude", f'{aq["peak_amplitude"]:.3f}')
                            if "rms_amplitude" in aq:
                                quality_cols[1].metric("RMS level", f'{aq["rms_amplitude"]:.4f}')
                            quality_cols = st.columns(2)
                            if "clipping_fraction" in aq:
                                quality_cols[0].metric("Samples clipped", f'{aq["clipping_fraction"] * 100:.2f}%')
                            if "silence_like" in aq:
                                quality_cols[1].metric(
                                    "Near-silent",
                                    "Yes" if aq["silence_like"] else "No",
                                )
                            if aq.get("silence_like"):
                                st.warning("This recording is near-silent or extremely quiet.")
                            if aq.get("clipping_fraction", 0) > 0.01:
                                st.warning(
                                    f'{aq["clipping_fraction"] * 100:.1f}% of samples may be clipped.'
                                )
                            st.markdown(
                                '<p class="audio-quality-note">Signal checks flag quiet or clipped '
                                "audio. They do not detect environmental noise or estimate SNR.</p></div>",
                                unsafe_allow_html=True,
                            )

                        if result["audio_truncated"]:
                            st.warning(
                                f"The CNN-LSTM processed {result['processed_duration_seconds']:.1f} s "
                                f"of the {result['audio_duration_seconds']:.1f} s recording "
                                f"(up to approximately {CNN_MAX_AUDIO_SECONDS:.1f} s at this input size). "
                                "The SVM Baseline instead summarizes features across the full recording."
                            )

                        seg_sec = 2.5
                        duration = len(audio) / sr
                        if duration > seg_sec * 2:
                            st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)
                            st.markdown(
                                '<div class="es-card"><div class="es-card-title">'
                                "Segment-level predictions · 2.5-second windows</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                "These are separate predictions for each complete window, not a "
                                "continuous emotion measurement."
                            )
                            with st.spinner("Analyzing audio segments…"):
                                timeline_html = segment_timeline_html(
                                    audio, sr, seg_sec, selected_model
                                )
                            if timeline_html:
                                st.markdown(timeline_html, unsafe_allow_html=True)
                            else:
                                st.info("The recording is too short to form complete analysis windows.")
                            st.markdown("</div>", unsafe_allow_html=True)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — NOISE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════

elif "📊" in page:
    st.markdown("## SVM Noise Evaluation")
    st.markdown(
        "Verified held-out evaluation of the SVM Baseline on clean audio and synthetic "
        "additive white noise. CNN-LSTM noise performance was not measured in this run; "
        "these results do not establish real-world noise robustness."
    )
    metrics_path = OUTPUT_DIR / "all_models_comparison.csv"
    predictions_path = OUTPUT_DIR / "all_models_predictions.csv"
    if not metrics_path.is_file():
        missing = [
            f"{path.parent.name}/{path.name}" for path in (
                metrics_path,
                predictions_path,
                OUTPUT_DIR / "test_manifest.csv",
                MODEL_DIR / "split_actors.json",
                MODEL_DIR / "svm_model.pkl",
                MODEL_DIR / "svm_scaler.pkl",
                MODEL_DIR / "svm_label_mapping.json",
            ) if not path.is_file()
        ]
        st.markdown(
            """<div class="setup-box">
            <h3>Held-out results are not available</h3>
            <p>This page requires the verified SVM metrics, per-record prediction
            evidence, held-out manifest, actor split, and SVM artifacts. No metrics
            are inferred from report prose or from the single-recording Noise Lab.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.info("Missing artifact(s): " + ", ".join(missing))
    else:
        try:
            metrics, test_actors, manifest_count, provenance = (
                _load_svm_noise_evaluation()
            )
        except (
            FileNotFoundError, OSError, ValueError, TypeError, KeyError,
            IndexError, OverflowError, AttributeError, UnicodeError,
            pd.errors.ParserError,
        ) as exc:
            st.error(
                "SVM noise-evaluation artifacts are incomplete, malformed, or inconsistent. "
                "No results are displayed."
            )
            with st.expander("Artifact validation details"):
                st.code(str(exc))
        else:
            st.caption(
                f"Model: {SVM_BASELINE} · "
                f"Test actors from split_actors.json: {', '.join(map(str, sorted(test_actors)))} · "
                f"Manifest recordings: {manifest_count:,} · SVM-only evaluation."
            )
            st.caption(
                "Per-record prediction evidence confirms the exact evaluated recordings "
                "and true labels across conditions. SHA-256 fingerprints identify the "
                "recorded input contents but are not a signed attestation of the run."
            )
            st.caption(
                f"Evaluation run: `{provenance['run_id']}` · "
                f"Started: {provenance['created_at_utc']}"
            )
            if not provenance["required_inputs_fingerprinted"]:
                st.warning(
                    "This run's provenance metadata lists missing required input "
                    "fingerprints; do not describe it as fully reproducible."
                )
            elif provenance["missing_context_inputs"]:
                st.info(
                    "Some contextual files were unavailable and are listed in the "
                    "run provenance metadata."
                )
            ordered = metrics.set_index("condition").reindex(
                [condition for condition in COND_ORDER if condition in metrics["condition"].values]
            ).dropna(subset=["accuracy"]).reset_index()
            metric_cols = st.columns(min(len(ordered), 4))
            for col, row in zip(metric_cols, ordered.itertuples(index=False)):
                label = COND_LABELS.get(row.condition, row.condition)
                delta = getattr(row, "accuracy_change_vs_clean", None)
                delta_text = (
                    f"{float(delta) * 100:+.1f} percentage points"
                    if delta is not None and pd.notna(delta) and row.condition != "clean"
                    else None
                )
                with col:
                    st.metric(
                        f"Accuracy · {label}",
                        f"{row.accuracy * 100:.2f}%",
                        delta=delta_text,
                        delta_color="normal",
                        help=f"{int(row.n_test_files)} held-out recordings for this condition.",
                    )

            chart_cols = st.columns(2)
            for column, metric, label in (
                (chart_cols[0], "accuracy", "Accuracy"),
                (chart_cols[1], "macro_f1", "Macro F1"),
            ):
                with column:
                    with plt.rc_context(MPL_STYLE):
                        fig, ax = plt.subplots(figsize=(6, 3.6))
                        labels = [COND_LABELS.get(value, value) for value in ordered["condition"]]
                        values = ordered[metric].to_numpy(dtype=float)
                        bars = ax.bar(
                            labels, values,
                            color=[ACCENT, ACCENT2, "#a89cff", "#4a5568"][:len(values)],
                            width=0.58,
                        )
                        ax.set_ylim(0, 1.05)
                        ax.set_ylabel(label)
                        ax.set_title(f"{label} by noise condition")
                        for bar, value in zip(bars, values):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                value + 0.015,
                                f"{value * 100:.1f}%",
                                ha="center", va="bottom", fontsize=8, color="#f3f4f8",
                            )
                        fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            display = ordered.copy()
            display["condition"] = display["condition"].map(
                lambda condition: COND_LABELS.get(condition, condition)
            )
            display = display.rename(columns={
                "condition": "Noise condition",
                "n_test_files": "Test recordings",
                "accuracy": "Accuracy",
                "macro_precision": "Macro precision",
                "macro_recall": "Macro recall",
                "macro_f1": "Macro F1",
                "accuracy_change_vs_clean": "Accuracy change vs clean",
                "macro_f1_change_vs_clean": "Macro F1 change vs clean",
            })
            for column in (
                "Accuracy", "Macro precision", "Macro recall", "Macro F1",
                "Accuracy change vs clean", "Macro F1 change vs clean",
            ):
                if column in display:
                    display[column] = pd.to_numeric(display[column], errors="coerce").map(
                        lambda value: f"{value * 100:+.2f}%" if "change" in column.lower()
                        else f"{value * 100:.2f}%"
                    )
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption(
                "Noise is synthetic white Gaussian noise. Results do not represent "
                "babble, music, reverberation, or all real-world recording conditions."
            )

            report_col, matrix_col = st.columns(2)
            with report_col:
                report_condition = st.selectbox(
                    "Per-class report condition",
                    options=ordered["condition"].tolist(),
                    format_func=lambda value: COND_LABELS.get(value, value),
                    key="robustness_report_condition",
                )
                report_path = OUTPUT_DIR / f"svm_classification_report_{report_condition}.csv"
                if report_path.is_file():
                    try:
                        report = pd.read_csv(report_path, index_col=0)
                        expected_report_columns = {"precision", "recall", "f1-score", "support"}
                        if report.empty or not expected_report_columns.issubset(report.columns):
                            raise ValueError("Missing precision, recall, f1-score, or support columns.")
                        for column in expected_report_columns:
                            report[column] = pd.to_numeric(report[column], errors="coerce")
                        if not np.isfinite(report[list(expected_report_columns)].to_numpy(dtype=float)).all():
                            raise ValueError("The report contains non-finite metric values.")
                        st.dataframe(
                            report.style.format(
                                {column: "{:.3f}" for column in expected_report_columns},
                                na_rep="—",
                            ),
                            use_container_width=True,
                        )
                    except (OSError, pd.errors.ParserError, ValueError) as exc:
                        st.warning(f"Could not display {report_path.name}: {exc}")
                else:
                    st.info(f"Per-class report not generated: {report_path.name}.")
            with matrix_col:
                matrix_path = OUTPUT_DIR / f"svm_confusion_matrix_{report_condition}.png"
                if matrix_path.is_file():
                    try:
                        if any(
                            matrix_path.stat().st_mtime_ns < path.stat().st_mtime_ns
                            for path in (metrics_path, predictions_path)
                        ):
                            raise ValueError(
                                "Image predates the validated metrics or prediction evidence."
                            )
                        image = plt.imread(matrix_path)
                        st.image(image, caption=f"Confusion matrix · {COND_LABELS[report_condition]}")
                    except (OSError, ValueError) as exc:
                        st.warning(f"Could not display {matrix_path.name}: {exc}")
                else:
                    st.info(f"Confusion matrix not generated: {matrix_path.name}.")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — SETUP GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

elif "ℹ️" in page:
    st.markdown("## Methodology & Setup")
    st.markdown(
        "A guide to the implemented data and inference workflows, the artifacts currently "
        "available in this checkout, and the steps needed to reproduce training and evaluation."
    )

    st.markdown("### Project workflow")
    workflow_cols = st.columns(3)
    workflow = (
        ("01 · Dataset", "RAVDESS audio-only speech files are discovered recursively under data/ and parsed from their seven-part filenames."),
        ("02 · Split", "train.py creates deterministic train, validation, and test groups by actor ID, keeping speakers separate across the three partitions."),
        ("03 · Features", "The original CNN-LSTM uses 40-coefficient MFCC sequences; the SVM uses pooled MFCC statistics; the improved CNN-LSTM adds delta features."),
        ("04 · Fit", "Separate scripts train the original CNN-LSTM, SVM Baseline, and improved CNN-LSTM. The SVM and improved variant reuse the saved actor split."),
        ("05 · Predict", "Audio Intelligence and Noise Lab can use a model only when its required saved inference artifacts are present."),
        ("06 · Evaluate", "evaluate.py evaluates the original CNN-LSTM; evaluate_all.py compares available models over clean and synthetic-noise test conditions."),
    )
    for index, (title, description) in enumerate(workflow):
        with workflow_cols[index % len(workflow_cols)]:
            st.markdown(
                f'<div class="es-card"><div class="es-card-title">{title}</div>'
                f'<p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    with st.expander("Feature pipelines and model roles", expanded=True):
        st.markdown(
            """
            - **Original CNN-LSTM:** extracts 40 MFCC coefficients per frame, normalizes
              each recording's coefficients, and pads or truncates to 200 frames before
              the CNN-LSTM. Its fixed input means uploaded recordings longer than about
              two seconds are truncated for this model.
            - **SVM Baseline:** summarizes each of the 40 MFCC coefficients with its
              mean, standard deviation, minimum, and maximum (160 values total). A
              `StandardScaler` is fitted on training data and reused for prediction.
              The saved SVM supports the current uploaded-audio and single-recording
              noise workflows.
            - **Improved CNN-LSTM:** the separate experiment uses MFCC, delta, and
              delta-delta features, with training-only synthetic-noise augmentation.
              Its weights are in `experiments/improved_cnn_lstm_run_01/`; it is not
              integrated into uploaded-audio prediction.
            """
        )
        st.caption(
            "Audio is loaded as mono at 16 kHz. CNN softmax scores and SVM probability "
            "estimates are not calibration-validated confidence probabilities."
        )

    st.markdown("### Noise demonstration and evaluation")
    noise_cols = st.columns(2)
    with noise_cols[0]:
        st.markdown(
            """<div class="es-card">
            <div class="es-card-title">Noise Lab · one recording</div>
            <p>The existing helper adds synthetic white Gaussian noise at a selected
            SNR. The page compares clean and noisy versions of the same uploaded
            recording using the same available model. This demonstrates a change in
            one prediction; it does not measure general robustness.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with noise_cols[1]:
        st.markdown(
            """<div class="es-card">
            <div class="es-card-title">Held-out evaluation · dataset results</div>
            <p>The evaluation scripts apply fixed, seeded noise to test-manifest
            recordings and calculate metrics and reports. Those outputs require
            trained weights and manifests. The artifact inventory below shows
            whether generated evaluation outputs are present in this checkout.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### Current artifact inventory")
    inventory_cols = st.columns(2)
    with inventory_cols[0]:
        st.markdown("**Dataset**")
        try:
            dataset_files, dataset_actors = dataset_inventory()
        except Exception as exc:
            st.error(f"Could not inspect dataset inventory: {exc}")
        else:
            if dataset_files:
                st.markdown(
                    f"{dataset_files:,} recognized RAVDESS speech recordings across "
                    f"{dataset_actors:,} actors are discoverable under `data/`."
                )
            else:
                st.info("No recognized RAVDESS speech recordings were found under `data/`.")

        st.markdown("**Prediction models**")
        for model_name in (SVM_BASELINE, ORIGINAL_CNN_LSTM):
            missing = missing_model_artifacts(model_name)
            if missing:
                st.markdown(
                    f"- **{model_name}:** unavailable; missing "
                    f"`{', '.join(missing)}`."
                )
            else:
                label = (
                    f"{model_name} (final selected model)"
                    if model_name == SVM_BASELINE else model_name
                )
                st.markdown(f"- **{label}:** available for prediction.")
        improved_model = (
            PROJECT_ROOT / "experiments" / "improved_cnn_lstm_run_01" / "improved_cnn_lstm.keras"
        )
        improved_mapping = (
            PROJECT_ROOT / "experiments" / "improved_cnn_lstm_run_01" / "label_mapping.json"
        )
        if improved_model.is_file() and improved_mapping.is_file():
            st.markdown(
                "- **Improved CNN-LSTM:** experimental weights and mapping are present "
                "in `experiments/improved_cnn_lstm_run_01/`; uploaded-audio inference is not integrated."
            )
        else:
            missing = [
                path.name for path in (improved_model, improved_mapping)
                if not path.is_file()
            ]
            st.markdown(
                f"- **Improved CNN-LSTM experiment:** missing `{', '.join(missing)}`."
            )

    with inventory_cols[1]:
        st.markdown("**Training and split artifacts**")
        for name in (
            "split_actors.json",
            "svm_config.json",
            "training_history.csv",
            "improved_training_history.csv",
        ):
            base = OUTPUT_DIR if name.endswith(".csv") else MODEL_DIR
            state = "present" if (base / name).is_file() else "not generated"
            st.markdown(f"- `{base.name}/{name}`: {state}.")

        st.markdown("**Dataset-level evaluation**")
        manifest_path = OUTPUT_DIR / "test_manifest.csv"
        st.markdown(
            f"- Held-out test manifest (`{manifest_path.name}`): "
            f"{'present' if manifest_path.is_file() else 'not generated'}."
        )
        st.markdown(
            f"- SVM-only noise evaluation: "
            f"{'verified' if evaluation_ready() else 'unavailable or failed validation'}."
        )
        st.markdown(
            f"- Final clean SVM/CNN-LSTM comparison: "
            f"{'verified' if final_clean_comparison_ready() else 'unavailable or failed validation'}."
        )

    st.markdown("### Reproduce the workflow")
    st.markdown(
        "Run these commands from the project root. They document the available scripts; "
        "training and evaluation are not started by this page."
    )
    st.code(
        "py -3.12 -m venv .venv\n"
        r".\.venv\Scripts\Activate.ps1" + "\n"
        "pip install -r requirements.txt\n"
        "python scripts/inspect_dataset.py",
        language="powershell",
    )
    st.markdown("**Train the original model and create the shared actor split first:**")
    st.code("python train.py", language="powershell")
    st.markdown(
        "**Optional model runs** (the latter two require the split from `train.py`; "
        "each uses separate model artifacts):"
    )
    st.code(
        "python train_svm.py\n"
        "python train_improved.py",
        language="powershell",
    )
    st.markdown("**Evaluation commands** (require the relevant trained weights and manifests):")
    st.code(
        "python evaluate.py       # Original CNN-LSTM, held-out clean/noise evaluation\n"
        "python evaluate_all.py   # Compare model artifacts available to the script",
        language="powershell",
    )
    st.markdown("**Launch the app or run the tests:**")
    st.code(
        "streamlit run app.py\n"
        "python -m pytest -q",
        language="powershell",
    )
    st.caption(
        "`evaluate_all.py` skips model configurations whose model-weight file is absent. "
        "For a model it does attempt, its manifest and remaining artifacts must also exist."
    )

    st.markdown("### Limitations and reproducibility notes")
    st.markdown(
        """
        - **Scores are not calibrated:** the displayed CNN softmax and SVM probability
          estimates rank model outputs but have not been validated as calibrated
          probabilities of correctness.
        - **Synthetic versus real noise:** the helper adds white Gaussian noise at
          target SNR; it does not simulate or validate performance on real-world
          babble, traffic, music, or reverberation.
        - **Weights and evaluation:** the SVM model, scaler, and label mapping are
          present. The original CNN-LSTM weights are unavailable in this checkout.
          Improved CNN-LSTM experiment weights are present under
          `experiments/improved_cnn_lstm_run_01/` but are not integrated into
          uploaded-audio inference.
        - **Pickle compatibility:** the SVM artifacts were serialized with
          scikit-learn 1.7.2, matching the project environment. Pickled estimators
          remain version-sensitive; this project does not silently rewrite them.
        - **Dataset scope:** the live inventory counts files matching this project's
          RAVDESS speech filename parser; it is not an independent audio-integrity
          audit or a measure of dataset quality.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE — NOISE DEMO
# ═══════════════════════════════════════════════════════════════════════════════

elif "🔬" in page or "Noise Demo" in page:
    st.markdown("## Noise Lab")
    st.markdown(
        "Compare one clean recording with a copy processed using synthetic white noise. "
        "Both versions use the same selected model and its existing inference pipeline. "
        "This interactive example is not a held-out-set robustness evaluation.",
        unsafe_allow_html=True,
    )
    if evaluation_ready():
        st.info(
            "Verified SVM-only held-out noise results are available on Noise Robustness. "
            "This page compares one uploaded recording and is not a substitute for that evaluation."
        )
    else:
        st.info(
            "SVM held-out noise results are currently unavailable or failed evidence validation. "
            "This page compares one uploaded recording only."
        )

    if not _models_avail:
        missing_artifacts = {
            model_name: missing_model_artifacts(model_name)
            for model_name in (ORIGINAL_CNN_LSTM, SVM_BASELINE)
        }
        st.warning(
            "No prediction model is available for this comparison. Missing artifacts: "
            + "; ".join(
                f"{name}: {', '.join(files) or 'none'}"
                for name, files in missing_artifacts.items()
            )
        )
    else:
        controls_col, model_col = st.columns([2, 1])
        with controls_col:
            snr_options = {
                "20 dB · Mild": 20,
                "10 dB · Moderate": 10,
                "0 dB · Strong": 0,
            }
            snr_choice = st.select_slider(
                "Synthetic white-noise target SNR",
                options=list(snr_options),
                value="10 dB · Moderate",
            )
            chosen_snr = snr_options[snr_choice]
        with model_col:
            default_model_index = (
                _models_avail.index(SVM_BASELINE) if SVM_BASELINE in _models_avail else 0
            )
            selected_model = st.selectbox(
                "Model for both recordings",
                options=_models_avail,
                index=default_model_index,
                key="noise_lab_model",
            )
        with st.expander("Model methods and limitations"):
            st.markdown(
                "**SVM Baseline:** MFCC mean, standard deviation, minimum, and maximum "
                "over the full recording, followed by its saved scaler. Its probability "
                "estimates are not calibration-validated."
            )
            st.markdown(
                f"**Original CNN-LSTM:** per-recording normalized MFCC sequence; "
                f"approximately {CNN_MAX_AUDIO_SECONDS:.1f} seconds maximum input. "
                "Longer recordings are truncated, and softmax scores are uncalibrated."
            )
            cnn_missing = missing_model_artifacts(ORIGINAL_CNN_LSTM)
            if cnn_missing:
                st.info(
                    "Original CNN-LSTM prediction is unavailable. Missing artifact(s): "
                    + ", ".join(cnn_missing)
                )

        st.caption(
            f"The existing `add_white_noise_at_snr` helper receives target SNR "
            f"{chosen_snr} dB and fixed seed 42. A single recording cannot establish model robustness."
        )
        uploaded_demo = st.file_uploader(
            "Choose a speech recording",
            type=["wav", "mp3", "flac", "ogg"],
            key="noise_demo_upload",
            help="Supported formats: WAV, MP3, FLAC, and OGG.",
        )

        if not uploaded_demo:
            st.markdown(
                """<div class="es-card audio-upload-card" style="text-align:center;">
                    <div style="font-size:1.8rem;margin-bottom:0.55rem;color:#c4bbff;">Single-recording comparison</div>
                    <div style="font-size:0.95rem;font-weight:550;color:#f3f4f8;">
                        Choose speech audio to compare clean and synthetic-noise predictions</div>
                    <div style="font-size:0.82rem;margin-top:0.45rem;color:#b0b8c8;">
                        WAV · MP3 · FLAC · OGG
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            suffix = Path(uploaded_demo.name).suffix or ".wav"
            demo_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    demo_path = Path(tmp.name)
                    tmp.write(uploaded_demo.getvalue())

                try:
                    demo_audio, demo_sr = librosa.load(
                        demo_path, sr=SAMPLE_RATE, mono=True
                    )
                    if demo_audio.size == 0:
                        raise ValueError("The decoded recording contains no audio samples.")
                except Exception as exc:
                    st.error(
                        "This recording could not be decoded. Check that it is a valid, "
                        "non-empty WAV, MP3, FLAC, or OGG file, then try another recording."
                    )
                    with st.expander("Technical details"):
                        st.code(str(exc))
                else:
                    import soundfile as sf

                    clean_bytes = uploaded_demo.getvalue()
                    clean_mime = {
                        ".wav": "audio/wav",
                        ".mp3": "audio/mpeg",
                        ".flac": "audio/flac",
                        ".ogg": "audio/ogg",
                    }.get(suffix.lower(), uploaded_demo.type or "application/octet-stream")
                    try:
                        noisy_audio = add_white_noise_at_snr(
                            demo_audio, snr_db=chosen_snr, seed=42
                        )
                        noisy_buffer = io.BytesIO()
                        sf.write(
                            noisy_buffer, noisy_audio, demo_sr,
                            format="WAV", subtype="PCM_16",
                        )
                        noisy_bytes = noisy_buffer.getvalue()
                    except Exception as exc:
                        st.error(
                            "Could not create the selected synthetic-noise version. "
                            "Check that the recording contains usable audio samples."
                        )
                        with st.expander("Noise-generation details"):
                            st.code(str(exc))
                        noisy_bytes = None

                    if noisy_bytes is not None:
                        st.markdown("### Recording comparison")
                        st.caption(
                            f"{uploaded_demo.name} · {len(demo_audio) / demo_sr:.2f} s · "
                            f"mono at {demo_sr:,} Hz · same {selected_model} pipeline for both versions"
                        )
                        clean_col, noisy_col = st.columns(2)
                        with clean_col:
                            st.markdown("#### Clean reference")
                            st.audio(clean_bytes, format=clean_mime)
                        with noisy_col:
                            st.markdown(f"#### Synthetic noise · target {chosen_snr} dB SNR")
                            st.audio(noisy_bytes, format="audio/wav")

                        with st.spinner(f"Comparing both recordings with {selected_model}…"):
                            try:
                                clean_result = predict_audio_bytes(
                                    clean_bytes, selected_model
                                )
                                noisy_result = predict_audio_bytes(
                                    noisy_bytes, selected_model
                                )
                            except FileNotFoundError as exc:
                                st.error(
                                    f"{selected_model} is unavailable because a required "
                                    "model artifact is missing. Restore the artifact and retry."
                                )
                                with st.expander("Missing artifact details"):
                                    st.code(str(exc))
                                clean_result = noisy_result = None
                            except Exception as exc:
                                st.error(
                                    f"{selected_model} could not compare these recordings. "
                                    "Try another supported speech file or verify the model artifacts."
                                )
                                with st.expander("Inference details"):
                                    st.code(str(exc))
                                clean_result = noisy_result = None

                        if clean_result is not None and noisy_result is not None:
                            clean_emotion = clean_result["predicted_emotion"]
                            noisy_emotion = noisy_result["predicted_emotion"]
                            changed = clean_emotion != noisy_emotion
                            result_cols = st.columns(2)
                            for column, label, result in (
                                (result_cols[0], "Clean prediction", clean_result),
                                (result_cols[1], f"Noisy prediction · {chosen_snr} dB target", noisy_result),
                            ):
                                emotion = result["predicted_emotion"]
                                with column:
                                    st.markdown(
                                        f"""<div class="es-card">
                                            <div class="es-card-title">{label}</div>
                                            <div class="audio-result">
                                                <div style="font-size:2.5rem;" aria-hidden="true">
                                                    {EMOTION_EMOJIS.get(emotion, "")}
                                                </div>
                                                <div>
                                                    <div class="audio-result-emotion">{emotion}</div>
                                                    <div class="audio-result-score">
                                                        {result["top_score"] * 100:.1f}% score
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="audio-score-note">{result["score_description"]}.
                                            Not a calibration-validated confidence probability.</div>
                                            <div class="es-card-title">Scores by emotion</div>
                                            {probability_bars_html(result["class_scores"])}
                                        </div>""",
                                        unsafe_allow_html=True,
                                    )

                            if changed:
                                st.warning(
                                    f"The predicted label changed from {clean_emotion} "
                                    f"to {noisy_emotion} for this recording at the selected "
                                    f"{chosen_snr} dB target."
                                )
                            else:
                                st.success(
                                    f"The predicted label stayed {clean_emotion} for this "
                                    f"recording at the selected {chosen_snr} dB target."
                                )
                            st.caption(
                                "This is a deterministic, single-recording illustration using "
                                "synthetic additive white noise (seed 42). It does not establish "
                                "held-out or real-world noise robustness."
                            )

                        aq = clean_result.get("audio_quality", {}) if clean_result else {}
                        if aq:
                            quality_cols = st.columns(4)
                            if "peak_amplitude" in aq:
                                quality_cols[0].metric("Clean peak", f'{aq["peak_amplitude"]:.3f}')
                            if "rms_amplitude" in aq:
                                quality_cols[1].metric("Clean RMS", f'{aq["rms_amplitude"]:.4f}')
                            if "clipping_fraction" in aq:
                                quality_cols[2].metric(
                                    "Clean clipped samples",
                                    f'{aq["clipping_fraction"] * 100:.2f}%',
                                )
                            if "silence_like" in aq:
                                quality_cols[3].metric(
                                    "Clean near-silent",
                                    "Yes" if aq["silence_like"] else "No",
                                )
                            st.caption(
                                "Signal checks are reported for the clean reference. "
                                "They are not environmental-noise or SNR measurements."
                            )
            finally:
                if demo_path is not None:
                    demo_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

elif "📈" in page or "Model Comparison" in page:
    st.markdown("## Model Comparison")
    st.markdown(
        "Clean held-out model comparison and SVM noise-condition results are separate "
        "experiments. They are validated and displayed independently."
    )
    st.markdown("### Final clean-audio comparison")
    try:
        final_comparison = _load_final_clean_comparison()
    except (
        FileNotFoundError, OSError, ValueError, TypeError, KeyError, IndexError,
        OverflowError, AttributeError, UnicodeError, pd.errors.ParserError,
    ) as exc:
        st.warning(
            "The final clean model comparison is unavailable because its evidence "
            "is missing or inconsistent."
        )
        with st.expander("Clean-comparison validation details"):
            st.code(str(exc))
    else:
        svm_result = final_comparison["models"]["svm_baseline"]
        improved_result = final_comparison["models"]["improved_cnn_lstm"]
        clean_cols = st.columns(2)
        for column, label, result in (
            (clean_cols[0], SVM_BASELINE, svm_result),
            (clean_cols[1], "Improved CNN-LSTM (experimental)", improved_result),
        ):
            with column:
                st.markdown(
                    f'<div class="es-card"><div class="es-card-title">{label}</div>'
                    f'<p>Accuracy: <strong>{result["accuracy"] * 100:.2f}%</strong><br>'
                    f'Macro-F1: <strong>{result["macro_f1"] * 100:.2f}%</strong></p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        actors = sorted(
            int(actor) for actor in final_comparison["provenance"]["test_actor_counts"]
        )
        st.caption(
            "Both models were evaluated on the same 300 held-out recordings "
            f"(test actors {', '.join(map(str, actors))}), using clean audio only. "
            "The CNN-LSTM is experimental; these results are not combined with the "
            "separate SVM noise-condition evaluation below."
        )
        st.caption(
            f"Clean comparison run timestamp: "
            f"{final_comparison['provenance']['run_timestamp_utc']} · "
            f"Manifest SHA-256: {final_comparison['provenance']['test_manifest_sha256']}"
        )

    st.markdown("### SVM noise-condition evaluation (separate experiment)")
    st.caption(
        "The table and charts below come from the SVM-only synthetic-noise evaluation; "
        "they are not an across-model ranking and contain no CNN-LSTM noise results."
    )
    comp_path = OUTPUT_DIR / "all_models_comparison.csv"
    predictions_path = OUTPUT_DIR / "all_models_predictions.csv"
    if not comp_path.is_file():
        missing = [
            f"{comp_path.parent.name}/{comp_path.name}",
            *(
                [f"{predictions_path.parent.name}/{predictions_path.name}"]
                if not predictions_path.is_file() else []
            ),
        ]
        for name in (
            "svm_model.pkl", "svm_scaler.pkl", "svm_label_mapping.json",
            "split_actors.json",
        ):
            base = MODEL_DIR
            if not (base / name).is_file():
                missing.append(f"{base.name}/{name}")
        if not (OUTPUT_DIR / "test_manifest.csv").is_file():
            missing.append(f"{OUTPUT_DIR.name}/test_manifest.csv")
        st.markdown(
            """<div class="setup-box">
            <h3>SVM noise-condition results are not available</h3>
            <p>The verified SVM metrics, per-record evidence, SVM artifacts, actor split,
            and held-out manifest are required for this separate noise experiment. The
            clean model comparison above is validated independently. This page does not
            train models or create evaluation outputs.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.info("Missing artifact(s): " + ", ".join(dict.fromkeys(missing)))
    else:
        try:
            comp_df = _read_metrics_csv(comp_path, comparison=True)
            sample_records = _read_evaluation_predictions(predictions_path)
            _validate_noise_seed_evidence(sample_records)
            provenance = _validate_run_provenance(
                comp_df, sample_records, "all_models_evaluation"
            )
            contexts = {}
            model_keys = comp_df["model_key"].drop_duplicates().tolist()
            expected_groups = [
                (model_key, condition)
                for model_key in model_keys
                for condition in sorted(ANALYTICS_CONDITIONS)
            ]
            actual_metric_groups = {
                (str(row.model_key), str(row.condition))
                for row in comp_df.itertuples(index=False)
            }
            if actual_metric_groups != set(expected_groups):
                raise ValueError(
                    "Comparison metrics must contain every model/noise-condition group."
                )
            if {str(record["model_key"]) for record in sample_records} != set(model_keys):
                raise ValueError(
                    "Prediction details do not contain the same model set as the metrics."
                )
            if not evaluation_samples_match(
                sample_records,
                ["model_key", "condition"],
                expected_groups=expected_groups,
            ):
                raise ValueError(
                    "Evaluated recording identities or true labels differ across "
                    "models or noise conditions."
                )
            _validate_sample_metric_counts(
                comp_df, sample_records, ("model_key", "condition")
            )
            if comp_path.stat().st_mtime_ns < predictions_path.stat().st_mtime_ns:
                raise ValueError(
                    "Comparison metrics predate per-record prediction evidence; "
                    "regenerate evaluation."
                )
            for model_key in model_keys:
                if model_key not in COMPARISON_MODEL_CONTEXTS:
                    raise ValueError(f"Unknown model_key in comparison CSV: {model_key}")
                context = COMPARISON_MODEL_CONTEXTS[model_key]
                if not comp_df.loc[comp_df["model_key"] == model_key, "model"].eq(
                    context["label"]
                ).all():
                    raise ValueError(
                        f"Model label does not match model_key {model_key} in {comp_path.name}."
                    )
                manifest_entries, manifest_count, test_actors, mapping = (
                    _validate_evaluation_context(
                        comp_path, model_key, context["manifest"]
                    )
                )
                _validate_prediction_label_mapping(
                    sample_records, model_key, mapping
                )
                if any(
                    not _sample_records_match_manifest(
                        sample_records,
                        group_columns=("model_key", "condition"),
                        group=(model_key, condition),
                        manifest_entries=manifest_entries,
                    )
                    for condition in sorted(ANALYTICS_CONDITIONS)
                ):
                    raise ValueError(
                        f"Prediction identities and labels for {model_key} do not "
                        "match its verified test manifest."
                    )
                contexts[model_key] = {
                    "manifest": manifest_entries,
                    "manifest_count": manifest_count,
                    "test_actors": test_actors,
                    "mapping": mapping,
                    "features": context["features"],
                    "label": context["label"],
                }
            if "accuracy_delta_vs_clean" in comp_df:
                delta_columns = [
                    column for column in (
                        "accuracy_delta_vs_clean", "macro_f1_delta_vs_clean",
                    ) if column in comp_df
                ]
                for column in delta_columns:
                    comp_df[column] = pd.to_numeric(comp_df[column], errors="coerce")
                    if (
                        not np.isfinite(comp_df[column].to_numpy(dtype=float)).all()
                        or ((comp_df[column] < -1) | (comp_df[column] > 1)).any()
                    ):
                        raise ValueError(
                            f"{comp_path.name} contains invalid {column} values."
                        )
        except (FileNotFoundError, ValueError) as exc:
            st.error(
                "Comparison artifacts are incomplete, malformed, stale, or inconsistent. "
                "No comparison results are displayed."
            )
            with st.expander("Artifact validation details"):
                st.code(str(exc))
        else:
            model_keys = comp_df["model_key"].drop_duplicates().tolist()
            comparable_counts = all(
                group["n_test_files"].nunique() == 1
                for _, group in comp_df.groupby("condition")
                if len(group) > 1
            )
            can_compare = (
                len(model_keys) > 1
                and comparable_counts
                and evaluation_samples_match(
                    sample_records,
                    ["model_key", "condition"],
                    expected_groups=expected_groups,
                )
            )
            if can_compare:
                st.success(
                    "Per-record evidence confirms the exact same evaluated recordings "
                    "and true labels for every model and noise condition. Feature "
                    "representations still differ as described in the table."
                )
            else:
                st.info(
                    "Results are shown by model, but a cross-model ranking/chart is withheld "
                    "because the verified manifests or exact evaluated "
                    "recording identities do not establish a like-for-like comparison."
                )

            st.caption(
                f"Evaluation run: `{provenance['run_id']}` · "
                f"Started: {provenance['created_at_utc']}"
            )
            if not provenance["required_inputs_fingerprinted"]:
                st.warning(
                    "This run's provenance metadata lists missing required input "
                    "fingerprints; do not describe it as fully reproducible."
                )
            elif provenance["missing_context_inputs"]:
                st.info(
                    "Some contextual files were unavailable and are listed in the "
                    "run provenance metadata."
                )

            display = comp_df.copy()
            display["feature_representation"] = display["model_key"].map(
                lambda key: contexts[key]["features"]
            )
            display["condition"] = display["condition"].map(
                lambda condition: COND_LABELS.get(condition, condition)
            )
            display = display.rename(columns={
                "model": "Model",
                "model_key": "Model key",
                "feature_representation": "Feature representation",
                "condition": "Noise condition",
                "n_test_files": "Test recordings",
                "accuracy": "Accuracy",
                "macro_precision": "Macro precision",
                "macro_recall": "Macro recall",
                "macro_f1": "Macro F1",
                "accuracy_delta_vs_clean": "Accuracy change vs clean",
                "macro_f1_delta_vs_clean": "Macro F1 change vs clean",
            })
            percent_columns = [
                column for column in (
                    "Accuracy", "Macro precision", "Macro recall", "Macro F1",
                    "Accuracy change vs clean", "Macro F1 change vs clean",
                ) if column in display
            ]
            for column in percent_columns:
                is_delta = "change" in column.lower()
                display[column] = pd.to_numeric(display[column], errors="coerce").map(
                    lambda value: f"{value * 100:+.2f}%" if is_delta
                    else f"{value * 100:.2f}%"
                )
            st.dataframe(display, use_container_width=True, hide_index=True)

            if can_compare:
                chart_conditions = [
                    condition for condition in COND_ORDER
                    if (comp_df["condition"] == condition).sum() == len(model_keys)
                ]
                if chart_conditions:
                    chart_columns = st.columns(2)
                    for column, metric, label in (
                        (chart_columns[0], "accuracy", "Accuracy"),
                        (chart_columns[1], "macro_f1", "Macro F1"),
                    ):
                        with column:
                            with plt.rc_context(MPL_STYLE):
                                fig, ax = plt.subplots(figsize=(7, 3.8))
                                x = np.arange(len(chart_conditions))
                                width = 0.8 / len(model_keys)
                                for index, key in enumerate(model_keys):
                                    rows = comp_df[comp_df["model_key"] == key].set_index("condition")
                                    values = [float(rows.loc[condition, metric]) for condition in chart_conditions]
                                    ax.bar(
                                        x + (index - (len(model_keys) - 1) / 2) * width,
                                        values,
                                        width=width * 0.9,
                                        label=contexts[key]["label"],
                                    )
                                ax.set_xticks(x)
                                ax.set_xticklabels([
                                    COND_LABELS.get(condition, condition)
                                    for condition in chart_conditions
                                ])
                                ax.set_ylim(0, 1.05)
                                ax.set_ylabel(label)
                                ax.set_title(f"{label} · verified shared conditions")
                                ax.legend(fontsize=8)
                                ax.grid(axis="y", alpha=0.25)
                                fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                else:
                    st.info("No noise condition has results for every listed model.")

            st.markdown("### Confusion matrix")
            matrix_columns = st.columns(2)
            with matrix_columns[0]:
                selected_matrix_model = st.selectbox(
                    "Model",
                    options=model_keys,
                    format_func=lambda key: contexts[key]["label"],
                    key="comparison_matrix_model",
                )
            available_matrix_conditions = comp_df.loc[
                comp_df["model_key"] == selected_matrix_model, "condition"
            ].tolist()
            with matrix_columns[1]:
                selected_matrix_condition = st.selectbox(
                    "Noise condition",
                    options=available_matrix_conditions,
                    format_func=lambda condition: COND_LABELS.get(condition, condition),
                    key="comparison_matrix_condition",
                )
            matrix_path = OUTPUT_DIR / (
                f"{selected_matrix_model}_confusion_matrix_{selected_matrix_condition}.png"
            )
            if not matrix_path.is_file():
                st.info(
                    f"Confusion matrix not generated: {matrix_path.name}. "
                    "Run `python evaluate_all.py` to generate it."
                )
            else:
                context = COMPARISON_MODEL_CONTEXTS[selected_matrix_model]
                freshness_paths = [
                    MODEL_DIR / name for name in context["artifacts"]
                ] + [
                    MODEL_DIR / "split_actors.json",
                    OUTPUT_DIR / context["manifest"],
                    predictions_path,
                ]
                try:
                    if any(
                        matrix_path.stat().st_mtime_ns < path.stat().st_mtime_ns
                        for path in freshness_paths
                    ):
                        raise ValueError(
                            "Image predates a current model, split, or test-manifest artifact."
                        )
                    matrix_image = plt.imread(matrix_path)
                    st.image(
                        matrix_image,
                        caption=(
                            f"{contexts[selected_matrix_model]['label']} · "
                            f"{COND_LABELS.get(selected_matrix_condition, selected_matrix_condition)}"
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "This image is the named output from evaluate_all.py. Its timestamp "
                        "is checked against the selected model, actor split, and test manifest. "
                        "Run-level fingerprints are available in the linked provenance JSON."
                    )
                except (OSError, ValueError) as exc:
                    st.warning(f"Could not display {matrix_path.name}: {exc}")

            st.caption(
                "Original CNN-LSTM and SVM Baseline use the shared test manifest; "
                "Improved CNN-LSTM uses its own manifest. Charts are withheld unless "
                "per-record prediction evidence confirms exact evaluated paths and "
                "semantic labels across models and conditions. The evaluation script "
                "binds metrics and per-record evidence to one run ID; SHA-256 input "
                "fingerprints are recorded in that run's provenance JSON."
            )
