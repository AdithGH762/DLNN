# Reports

The [EmotionSense project report](project_report.md) documents the RAVDESS
speech-emotion project, its actor-independent train/validation/test split,
SVM and CNN-LSTM approaches, verified clean and synthetic-noise results,
limitations, and experiment provenance.

Key supporting evidence:

- Clean baseline SVM vs. improved CNN-LSTM comparison:
  [`experiments/final_model_comparison/final_comparison.json`](../experiments/final_model_comparison/final_comparison.json)
- SVM clean and AWGN evaluation metrics:
  [`outputs/all_models_comparison.csv`](../outputs/all_models_comparison.csv)
- Run-specific SVM noise evaluation provenance:
  [`outputs/all_models_evaluation_run_695324b2e90a4af5b72034c5b95f5ebb.json`](../outputs/all_models_evaluation_run_695324b2e90a4af5b72034c5b95f5ebb.json)
- Improved CNN-LSTM training summary:
  [`experiments/improved_cnn_lstm_run_01/experiment_summary.json`](../experiments/improved_cnn_lstm_run_01/experiment_summary.json)
- Separate validation-tuned SVM experiment summary:
  [`experiments/svm_rbf_tuning/experiment_summary.json`](../experiments/svm_rbf_tuning/experiment_summary.json)

The original CNN-LSTM historical checkpoint/results and CNN-LSTM noise
metrics are not presented as verified outcomes. See the project report for
the evidence scope and remaining limitations.
