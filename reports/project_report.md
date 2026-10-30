# EmotionSense: Noise-Aware Speech Emotion Recognition Using CNN-LSTM

## Abstract

EmotionSense is a speech emotion recognition (SER) project using the audio-only
speech subset of RAVDESS. This report presents a baseline radial-basis-function
SVM, an improved CNN-LSTM using MFCC delta features, a validation-only SVM
hyperparameter experiment, and a controlled synthetic-noise evaluation. The
final clean comparison evaluates the saved SVM baseline and improved CNN-LSTM
on the same 300 actor-independent test recordings, drawn from actors 10, 16,
17, 19, and 20 (60 recordings per actor). The SVM achieved 47.00% accuracy
(141/300; macro-F1 0.4556); the improved CNN-LSTM achieved 18.33% accuracy
(55/300; macro-F1 0.1663). In the separate SVM noise evaluation, accuracy
decreased from 47.00% on clean audio to 13.33% at 0 dB SNR. These results show
limited clean classification performance for the CNN-LSTM experiment and
substantial SVM degradation under stronger synthetic noise. The two clean-test
pipelines differ in their feature representations and training procedures, so
their difference cannot be attributed to one design choice. The original
CNN-LSTM checkpoint and its historical results remain unavailable and are not
treated as measured results here.

---

## 1. Problem Statement and Objectives

Speech emotion recognition predicts an emotion label from acoustic properties
of a speech recording. A useful system should generalize to speakers not seen
during training and should be evaluated under clearly specified recording
conditions. Synthetic noise provides a controlled robustness test, but it does
not reproduce the full range of real acoustic environments.

The project objectives are:

- Prepare and inspect RAVDESS audio-only speech recordings and their eight
  filename-encoded emotion classes.
- Establish an MFCC-statistics SVM baseline.
- Implement CNN-LSTM sequence models, including an improved version using
  temporal MFCC derivatives and training-only noise augmentation.
- Preserve actor separation between training, validation, and held-out test
  data.
- Compare the SVM baseline and improved CNN-LSTM on the same clean test
  recordings.
- Measure SVM performance on clean and synthetic AWGN-corrupted test audio.
- Report results with per-class metrics and provenance, and distinguish
  measured results from hypotheses and unavailable historical results.

## 2. Dataset and Emotion Classes

**Dataset:** Ryerson Audio-Visual Database of Emotional Speech and Song
(RAVDESS), audio-only speech modality (filename modality code `03`).

The local filename inventory contains 1,440 matching speech-recording
filenames from 24 actors, 60 per actor. This is a filename count: it does not
by itself prove that every audio file is readable or that every file completed
preprocessing in every experiment.

| Emotion | Inventory filenames | Test support |
|---|---:|---:|
| Angry | 192 | 40 |
| Calm | 192 | 40 |
| Disgust | 192 | 40 |
| Fearful | 192 | 40 |
| Happy | 192 | 40 |
| Neutral | 96 | 20 |
| Sad | 192 | 40 |
| Surprised | 192 | 40 |
| **Total** | **1,440** | **300** |

Neutral has fewer recordings in the RAVDESS speech subset than the other
classes. The held-out test set therefore has 20 neutral examples and 40
examples for each of the other seven emotions.

## 3. Actor-Independent Experimental Split

The saved `models/split_actors.json` defines mutually disjoint actor
partitions:

| Partition | Actor IDs | Number of actors | Inventory/recordings used |
|---|---|---:|---:|
| Training | 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 21, 22, 23 | 14 | 840 |
| Validation | 7, 8, 11, 18, 24 | 5 | 300 |
| Test | 10, 16, 17, 19, 20 | 5 | 300 |

The final test manifest contains 300 unique recordings from exactly the five
test actors, with 60 recordings per actor. The improved CNN-LSTM training
summary records 840 training and 300 validation recordings used. The SVM
noise-evaluation and final clean-comparison evidence each report 300 test
recordings. The actor-level split is designed to measure performance on
unseen speakers; it does not by itself ensure generalization to other datasets
or natural conversational speech.

## 4. Audio Preprocessing and Feature Extraction

Audio is loaded as mono at 16 kHz. The project uses librosa for loading and
feature extraction.

### 4.1 Original CNN-LSTM features

The original CNN-LSTM pipeline uses 40 MFCC coefficients with `n_fft=512`,
`hop_length=160`, and `n_mels=64`. Sequences are padded or truncated to 200
frames and normalized per recording, producing a `(200, 40, 1)` tensor.

### 4.2 Improved CNN-LSTM features

The improved pipeline concatenates the 40 MFCC coefficients with first- and
second-order temporal derivatives (delta and delta-delta), yielding 120
coefficients per frame. After padding or truncation to 200 frames, each
recording has shape `(200, 120, 1)`. These derivatives represent changes in
the cepstral features over time; their effect cannot be isolated from the
other training changes in this experiment.

### 4.3 SVM features

The SVM computes 40 MFCC coefficients and summarizes each coefficient over
time by its mean, standard deviation, minimum, and maximum. This gives a
160-dimensional feature vector. A `StandardScaler` is fitted using training
data and applied before SVM prediction.

## 5. Models and Training Procedures

### 5.1 SVM baseline

The baseline is an RBF-kernel SVC using `class_weight='balanced'`. Its feature
input is the 160-dimensional MFCC statistics vector. The saved baseline
configuration uses `C=1` and `gamma='scale'`. The scaler is fitted on
training data. The baseline's final clean and noise results are tied to the
saved model artifacts and test manifest by the evaluation provenance files.

### 5.2 Improved CNN-LSTM

The improved model processes the `(200, 120, 1)` MFCC/delta/delta-delta
sequence. Its CNN-LSTM architecture is:

```text
Input (200, 120, 1)
  ├─ Reshape to (200, 120)
  ├─ Conv1D(64, kernel size 5, same padding, ReLU)
  ├─ Batch normalization
  ├─ Max pooling (2)
  ├─ Conv1D(96, kernel size 3, same padding, ReLU)
  ├─ Batch normalization
  ├─ Max pooling (2)
  ├─ LSTM(64, dropout 0.2)
  ├─ Dense(64, ReLU)
  ├─ Dropout(0.35)
  └─ Dense(8, softmax)
```

The saved isolated training experiment used Adam with initial learning rate
`5e-4`, sparse categorical cross-entropy, batch size 16, balanced class
weights, and a maximum of 80 epochs. Training-only augmentation added AWGN
with probability 0.5 at a randomly selected SNR between 5 and 30 dB.
Validation audio was not augmented. EarlyStopping monitored `val_loss`, used
patience 15, and restored the best weights. The best-validation checkpoint
was selected by minimum validation loss.

### 5.3 Original CNN-LSTM status

The project also contains source code for an original CNN-LSTM using 40 MFCCs
per frame. Its historical checkpoint and corresponding training and test
results are not available in the evidence used for this report. Therefore,
no original CNN-LSTM accuracy, per-class score, or noise result is presented
as verified.

## 6. Experimental Design

### 6.1 Final clean comparison

The saved final comparison evaluated the baseline SVM and improved CNN-LSTM
on the exact same 300 rows of `outputs/test_manifest.csv`. Both models used
the same recording identities and ground-truth labels. The SVM used its
associated scaler; the CNN-LSTM used the training-time MFCC/delta
preprocessing with augmentation disabled. This is a clean-audio comparison.

The comparison is descriptive, not a controlled ablation: the models differ
in feature representation, architecture, class weighting, and training
procedure. It cannot identify the causal contribution of any one difference.

### 6.2 SVM noise evaluation

The SVM was evaluated on clean audio and on copies of the same held-out
recordings with synthetic additive white Gaussian noise at 20, 10, and 0 dB
SNR. The run recorded 300 evaluated recordings per condition. Noise was
applied during evaluation; the source recordings were not altered. CNN-LSTM
noise metrics were not produced by this verified run.

### 6.3 Separate SVM hyperparameter experiment

A separate RBF-SVM experiment evaluated 16 combinations of `C` and `gamma`
using training actors for fitting and validation actors for selection. It
selected `C=10`, `gamma=0.001` by validation macro-F1, with validation
accuracy as a tie-breaker. The held-out test set was not used for model
selection. A later one-shot held-out comparison of that tuned candidate with
the baseline is reported separately below; test results are not a tuning
criterion.

## 7. Results

### 7.1 Final clean test comparison

Both models were evaluated on the same 300 clean test recordings from actors
10, 16, 17, 19, and 20.

| Model | Correct | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| SVM baseline | 141/300 | 47.00% | 48.17% | 45.94% | 0.4556 | 0.4632 |
| Improved CNN-LSTM | 55/300 | 18.33% | 18.97% | 17.81% | 0.1663 | 0.1695 |

Per-emotion metrics for this same clean comparison are:

| Emotion | Support | SVM precision | SVM recall | SVM F1 | Improved CNN-LSTM precision | Improved CNN-LSTM recall | Improved CNN-LSTM F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Angry | 40 | 0.700 | 0.700 | 0.700 | 0.121 | 0.100 | 0.110 |
| Calm | 40 | 0.559 | 0.475 | 0.514 | 0.179 | 0.125 | 0.147 |
| Disgust | 40 | 0.338 | 0.575 | 0.426 | 0.259 | 0.175 | 0.209 |
| Fearful | 40 | 0.403 | 0.625 | 0.490 | 0.224 | 0.325 | 0.265 |
| Happy | 40 | 0.421 | 0.200 | 0.271 | 0.155 | 0.400 | 0.224 |
| Neutral | 20 | 0.400 | 0.300 | 0.343 | 0.143 | 0.100 | 0.118 |
| Sad | 40 | 0.355 | 0.275 | 0.310 | 0.222 | 0.050 | 0.082 |
| Surprised | 40 | 0.677 | 0.525 | 0.592 | 0.214 | 0.150 | 0.176 |

The full-precision classification reports, confusion matrices, and
per-record predictions are saved in
`experiments/final_model_comparison/final_comparison.json` and
`experiments/final_model_comparison/final_predictions.csv`. Confusion-matrix
images are saved alongside them.

### 7.2 SVM clean and synthetic-noise results

| Evaluation condition | Accuracy | Macro-F1 | Evaluated recordings |
|---|---:|---:|---:|
| Clean | 47.00% | 0.4556 | 300 |
| 20 dB SNR | 23.67% | 0.1778 | 300 |
| 10 dB SNR | 14.00% | 0.0415 | 300 |
| 0 dB SNR | 13.33% | 0.0294 | 300 |

Noise robustness was evaluated, but the measured SVM results do **not** show
strong robustness: accuracy and macro-F1 fall substantially as the injected
noise level increases. These measurements apply to the specified synthetic
AWGN conditions and this held-out RAVDESS subset; they do not establish
performance under all real-world noise.

### 7.3 Improved CNN-LSTM training observations

The training run used 840 training and 300 validation recordings and completed
20 epochs. Minimum validation loss selected epoch 5 (`val_loss` 1.9425), at
which validation accuracy was 23.67%. The maximum observed validation
accuracy was 28.33% at epoch 8; this was **not** the selected checkpoint
because selection monitored validation loss. Maximum training accuracy was
97.98% at epoch 17. At the final epoch (20), training accuracy was 97.74%
and validation accuracy was 26.33%. These are distinct observations. The
large train/validation gap is evidence of substantial overfitting or poor
generalization in this run; it does not by itself identify the cause.

### 7.4 Separate SVM tuning results

| Configuration | Validation accuracy | Validation macro-F1 | Held-out test accuracy | Held-out test macro-F1 |
|---|---:|---:|---:|---:|
| Baseline (`C=1`, `gamma=scale`) | 50.00% | 0.4802 | 47.00% | 0.4556 |
| Selected candidate (`C=10`, `gamma=0.001`) | 53.33% | 0.5244 | 44.00% | 0.4174 |

The candidate was selected on validation macro-F1 from 16 configurations.
Its held-out result was lower than the baseline in this separate test
comparison. This result is reported as a final comparison only and must not
be used to revise the selected configuration or otherwise tune against the
test set. This SVM-versus-SVM experiment is separate from the SVM-versus-CNN
comparison in §7.1. The tuned-SVM held-out comparison run ID is
`3c4d522f5dff4246a5eaed5b3ec9c525`; it is distinct from the validation-tuning
experiment and the main SVM noise-evaluation run.

## 8. Discussion

On the shared clean test cohort, the SVM baseline outperformed the improved
CNN-LSTM on accuracy and macro-F1. The CNN-LSTM’s training history shows
training accuracy rising substantially above validation accuracy, while the
best checkpoint by validation loss had only 23.67% validation accuracy.
Together, these measurements indicate that this improved CNN-LSTM run did
not generalize well to the held-out actors.

The SVM noise results show a sharp degradation under stronger AWGN. Although
the system includes a controlled noise-evaluation workflow, the observed
performance does not justify describing the SVM as robust to strong noise.
There are no verified CNN-LSTM noise results here, so no noise-condition
ranking between the models can be made.

The two clean-test models do not differ in only one factor. In addition to
using pooled MFCC statistics versus temporal MFCC/delta sequences, they use
different model families and training procedures. The measured difference
therefore cannot establish that delta features, noise augmentation, class
weighting, or architecture alone caused the CNN-LSTM result.

The validation-selected SVM candidate scored better than the baseline on
validation but lower on the separate held-out comparison. This illustrates
why model selection must use training/validation data and why held-out results
should be reported once rather than used for further tuning.

## 9. Limitations

1. **Dataset scope:** RAVDESS contains acted speech. Results from this dataset
   do not establish performance on spontaneous emotion, other languages,
   populations, microphones, or recording environments.
2. **Filename inventory versus usable audio:** the 1,440 count is based on
   matching filenames. It does not certify the readability of every file in
   the full inventory.
3. **Synthetic noise:** AWGN at fixed SNRs is a controlled perturbation and
   does not model all real noises, including overlapping speech, traffic,
   music, reverberation, or device-specific effects.
4. **Noise model scope:** the verified dataset-level noise metrics are for the
   SVM only. CNN-LSTM performance under those noise conditions is unknown.
5. **CNN-LSTM generalization:** the improved CNN-LSTM run shows a substantial
   training/validation gap and low held-out clean performance. This report
   does not attribute that gap to a single cause.
6. **Non-ablated model comparison:** features and training procedures differ,
   so their individual causal effects were not isolated.
7. **Original CNN-LSTM evidence:** its historical checkpoint and evaluation
   results are unavailable and cannot be compared as verified outcomes.
8. **Single held-out cohort:** metrics are measured on one actor-level split.
   No confidence intervals or repeated actor-independent splits are reported.
9. **SVM tuning:** only the stated 16-configuration RBF grid was tested. Its
   selected candidate did not exceed the baseline on the final held-out
   comparison.
10. **Deployment:** these experiments do not establish production readiness,
    calibrated confidence, or real-world reliability.

## 10. Reproducibility and Provenance

### 10.1 Final clean comparison

The saved clean comparison timestamp is
`2026-10-30T06:39:32.376760+00:00`. It used the manifest
`outputs/test_manifest.csv`, SHA-256
`9481ae8073e24b49c85c64812151fe978c4b2c74481190900b2e182a196478c0`,
and the actor split listed in §3. Its provenance file records SHA-256
fingerprints for the SVM model, scaler, mapping, improved CNN-LSTM weights
and mapping, manifest, and split. Full values are retained in
`experiments/final_model_comparison/final_comparison.json`.

The recorded environment was Python 3.12.10, TensorFlow 2.18.1, scikit-learn
1.7.2, NumPy 2.0.2, pandas 2.3.3, librosa 0.11.0, matplotlib 3.11.2, and
joblib 1.6.0.

### 10.2 SVM noise evaluation

The verified noise run ID is
`695324b2e90a4af5b72034c5b95f5ebb`. Aggregate metrics are in
`outputs/all_models_comparison.csv`; per-record evidence is in
`outputs/all_models_predictions.csv`; run-specific provenance is in
`outputs/all_models_evaluation_run_695324b2e90a4af5b72034c5b95f5ebb.json`.
The evidence records 300 evaluated SVM recordings in each of the four
conditions and identifies the five held-out actors. The run provenance
contains the evaluation settings, software versions, and input fingerprints.

### 10.3 Training and tuning artifacts

The improved CNN-LSTM training history and summary are saved under
`experiments/improved_cnn_lstm_run_01/`. They record the actor IDs, numbers
of training and validation recordings, preprocessing and augmentation
settings, software versions, and selected epoch.

The separate SVM tuning grid and summary are under
`experiments/svm_rbf_tuning/`. Its final held-out metrics and provenance are
under `experiments/svm_rbf_tuning/final_test/`, including the fact that the
test set was not used for selection. The final model-comparison predictions
and metrics are likewise preserved in their experiment output directory.

These artifacts support the results reported here, but do not make the
experiments universally reproducible: reproducing them also requires the
specified data files, compatible dependencies, and the same saved models and
configuration inputs. The original CNN-LSTM historical results cannot be
reproduced from the available checkpoint evidence.

## 11. Conclusion

This project evaluated RAVDESS speech emotion classification using an
MFCC-statistics SVM and an improved MFCC/delta CNN-LSTM. On the same
300-record, actor-independent clean test set, the SVM achieved 47.00%
accuracy and macro-F1 0.4556, while the improved CNN-LSTM achieved 18.33%
accuracy and macro-F1 0.1663. These are measured results for this test cohort,
not evidence of high accuracy or real-world readiness.

The improved CNN-LSTM training history showed a large training/validation
gap, and its checkpoint was selected at epoch 5 by minimum validation loss.
In the separate SVM noise experiment, accuracy fell to 13.33% at 0 dB SNR,
so strong noise robustness was not achieved. The validation-tuned SVM
candidate also scored below the baseline on the separate held-out comparison.
The results support reporting the SVM as the stronger of these two measured
clean-test pipelines, but do not establish which individual architectural or
feature choice caused the difference. The original CNN-LSTM and CNN-LSTM
noise performance remain unverified.

## 12. References

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual
  Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5),
  e0196391. https://doi.org/10.1371/journal.pone.0196391
- RAVDESS dataset: https://zenodo.org/records/1188976
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
  *Neural Computation*, 9(8), 1735–1780.
- Davis, S., & Mermelstein, P. (1980). Comparison of parametric
  representations for monosyllabic word recognition in continuously spoken
  sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*,
  28(4), 357–366.
- Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.
