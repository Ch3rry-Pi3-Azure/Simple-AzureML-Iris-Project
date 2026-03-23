"""
Shared artifact filenames used across local and Azure ML workflows.

This module centralises the filenames written by the local training
script and the Azure ML pipeline steps. Keeping these names in one
place reduces duplication and lowers the risk of local and remote
workflows drifting apart over time.
"""

from __future__ import annotations

from typing import Final


# Core structured metric outputs
METRICS_JSON: Final[str] = "metrics.json"
BEST_PARAMS_JSON: Final[str] = "best_params.json"
CV_RESULTS_CSV: Final[str] = "cv_results.csv"
GRID_SEARCH_SUMMARY_JSON: Final[str] = "grid_search_summary.json"

# Text and JSON reports
CLASSIFICATION_REPORT_TXT: Final[str] = "classification_report.txt"
CLASSIFICATION_REPORT_JSON: Final[str] = "classification_report.json"
CONFUSION_MATRIX_JSON: Final[str] = "confusion_matrix.json"

# Plot outputs
CONFUSION_MATRIX_PNG: Final[str] = "confusion_matrix.png"
CLASSIFICATION_REPORT_PNG: Final[str] = "classification_report.png"
ROC_CURVE_PNG: Final[str] = "roc_curve.png"
LEARNING_CURVE_PNG: Final[str] = "learning_curve.png"
OOB_ERROR_CURVE_PNG: Final[str] = "oob_error_curve.png"
