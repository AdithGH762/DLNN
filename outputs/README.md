Evaluation metrics, dataset summaries, charts, and CSV reports are generated here.

The evaluation scripts also write per-record evidence:

- `noise_evaluation_predictions.csv`: original CNN-LSTM results for each manifest
  recording and noise condition.
- `all_models_predictions.csv`: results for each available model, recording, and
  noise condition.

These files contain `run_id`, `model_key`, `condition`, `path`, integer `y_true`, semantic
`true_label`, integer `y_pred` and semantic `predicted_label` for evaluated rows,
`status`, `error`, `noise_base_seed`, and `noise_seed`. Seed fields are blank for
the clean condition; noisy rows store the evaluator's base seed and the exact
SHA-256-derived seed passed to the AWGN generator. Status is `evaluated`,
`preprocessing_error`, or `inference_error`. A failed row has no prediction.
Each row also has `recording_sha256`, the content fingerprint of its source audio
when that file was available to fingerprint.
Aggregate robustness/comparison outputs are generated only when all required
groups have matching evaluated recording identities and true labels.
The Streamlit analytics pages require this evidence; older aggregate CSVs without
the corresponding per-record file are not treated as verified comparisons.

Each new evaluation also writes a uniquely named provenance JSON sidecar:
`noise_evaluation_run_<run_id>.json` or
`all_models_evaluation_run_<run_id>.json`. It records the UTC creation time,
completion status, software versions, available Git revision, evaluation settings,
and SHA-256 fingerprints of available model weights, scalers, mappings, manifests,
configuration/context files, and relevant source files. Missing inputs are listed
without a fabricated fingerprint. The `run_id` is present in both aggregate metric
rows and per-record evidence; each recording fingerprint is repeated in run metadata.
The app requires all three artifacts and recording fingerprints to agree before
displaying results. `required_inputs_fingerprinted` distinguishes complete required
input coverage from runs with missing required inputs, while contextual absences
are listed separately.
