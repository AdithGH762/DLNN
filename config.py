from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# RAVDESS Audio_Speech_Actors_01-24.zip extracts Actor_* folders directly into data/.
# If your extraction placed them inside data/ravdess/ instead, adjust this line.
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

SAMPLE_RATE = 16_000
N_MFCC = 40
MAX_FRAMES = 200
RANDOM_SEED = 42

# RAVDESS modality codes. "03" = audio-only speech. "02" = audio-only song.
# We only train on speech files.
SPEECH_MODALITY = "03"

# RAVDESS speech filename emotion codes:
# 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised
EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
