| Metric | Value |
| --- | --- |
| Samples | 36382 |
| Positive samples | 3095 |
| Positive rate (%) | 8.507 |
| Raw PR-AUC | 0.2776 |
| Raw ROC-AUC | 0.8043 |
| Calibrated Brier score | 0.0812 |
| Calibrated log loss | 0.2754 |
| Calibrated precision | 0.1818 |
| Calibrated recall | 0.6384 |
| Calibrated F1 | 0.2830 |

Note: Each held-out satellite fold used its own Platt calibrator. Pooled calibrated Brier score and log loss summarize per-sample probability errors, but calibrated PR-AUC and ROC-AUC are omitted because fold-specific calibration can change score ordering across folds. Raw pooled ranking metrics are reported instead.
