from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, OUTPUT_DIR
from src.dataset import discover_ravdess, dataset_summary


def main():
    df = discover_ravdess(DATA_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print(f"No valid RAVDESS speech files found under: {DATA_DIR}")
        print("Extract Audio_Speech_Actors_01-24.zip so that Actor_01/, Actor_02/, ...")
        print(f"... are placed directly inside: {DATA_DIR}")
        print("Expected filenames resemble: 03-01-05-01-02-01-12.wav")
        return
    by_emotion, by_actor = dataset_summary(df)
    df.to_csv(OUTPUT_DIR / "dataset_manifest.csv", index=False)
    by_emotion.to_csv(OUTPUT_DIR / "files_by_emotion.csv", index=False)
    by_actor.to_csv(OUTPUT_DIR / "files_by_actor.csv", index=False)
    print(f"Valid RAVDESS audio-only speech files discovered: {len(df)}")
    print(f"Actors: {df.actor_id.nunique()}  |  Emotions: {df.emotion.nunique()}")
    print("\nFiles by emotion:")
    print(by_emotion.to_string(index=False))
    print("\nFiles by actor:")
    print(by_actor.to_string(index=False))
    print(f"\nManifest saved to {OUTPUT_DIR / 'dataset_manifest.csv'}")
    print("Note: this script reports actual discovered classes; it does not fabricate missing classes.")


if __name__ == "__main__":
    main()
