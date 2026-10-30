# EmotionSense

EmotionSense is a Streamlit speech-emotion recognition project using the
audio-only speech subset of RAVDESS. It explores eight filename-encoded
emotions: angry, calm, disgust, fearful, happy, neutral, sad, and surprised.

## Models and evaluation

- **Baseline SVM:** summarizes 40 MFCC coefficients using mean, standard
  deviation, minimum, and maximum (160 features total), then applies a
  training-fitted scaler and an RBF SVM.
- **Improved CNN-LSTM:** uses 40 MFCC coefficients plus delta and
  delta-delta features, arranged as 200-frame sequences with 120 feature
  coefficients per frame.
- **Actor-independent split:** 14 training actors, 5 validation actors, and
  5 held-out test actors. The final clean comparison uses 300 recordings:
  actors 10, 16, 17, 19, and 20 (60 recordings each).

On the same clean held-out recordings, the baseline SVM achieved **47.00%
accuracy** and **45.56% macro-F1**. The improved CNN-LSTM achieved **18.33%
accuracy** and **16.63% macro-F1**. The baseline SVM is the selected model
based on this measured clean-test comparison. Its result is limited and is
not evidence of high accuracy or production readiness.

Verified SVM accuracy under synthetic additive white Gaussian noise (AWGN):

| Condition | Accuracy |
|---|---:|
| Clean | 47.00% |
| 20 dB SNR | 23.67% |
| 10 dB SNR | 14.00% |
| 0 dB SNR | 13.33% |

Performance degrades substantially as synthetic noise increases. These
results do not establish robustness to real-world noise. The verified
noise-evaluation run covers the SVM only; CNN-LSTM noise performance is not
reported. The two clean-test pipelines also differ in features and training
procedures, so their comparison does not isolate a single causal factor.

## Setup and launch

Use Python 3.12, which is within the versions targeted by this project.
From PowerShell in the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app entry point is `app.py`; dependency constraints are defined in
`requirements.txt`. Dataset files and trained artifacts are not installed by
these commands. For dataset setup, obtain the RAVDESS speech archive from the
[official RAVDESS record](https://zenodo.org/records/1188976) and follow its
terms.

## Results and report

- [Detailed academic report](reports/project_report.md)
- [Clean SVM vs. improved CNN-LSTM metrics and provenance](experiments/final_model_comparison/final_comparison.json)
- [Clean comparison summary](experiments/final_model_comparison/comparison_summary.json)
- [Per-record clean predictions](experiments/final_model_comparison/final_predictions.csv)
- [SVM clean and noise metrics](outputs/all_models_comparison.csv)
- [SVM noise-run provenance](outputs/all_models_evaluation_run_695324b2e90a4af5b72034c5b95f5ebb.json)
- [SVM per-emotion report (clean)](outputs/svm_classification_report_clean.csv)

The detailed report describes the actor split, methods, per-class results,
training observations, limitations, and experiment provenance.
