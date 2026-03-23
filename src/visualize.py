"""
Visualization helpers for training and evaluation artifacts.

This module groups the plotting utilities used by both the local
training workflow and the Azure ML pipeline steps. The plots are
written to disk as static images so they can be viewed in Azure ML
job outputs, downloaded locally, or archived alongside other run
artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so plot generation works in headless
# environments such as Azure ML jobs and CI runners.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize


# Canonical human-readable class names used in plots.
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def save_confusion_matrix_plot(confusion_matrix, output_path: Path) -> None:
    """
    Render and save a confusion-matrix plot.

    Parameters
    ----------
    confusion_matrix :
        Square matrix containing true-vs-predicted class counts.

    output_path : Path
        Filesystem path where the PNG should be written.

    Returns
    -------
    None
        The function writes the image to disk and does not return a
        value.
    """

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_classification_report_heatmap(
    classification_report_dict: dict,
    output_path: Path,
) -> None:
    """
    Render and save a classification-report heatmap.

    Parameters
    ----------
    classification_report_dict : dict
        Structured classification report produced by
        ``sklearn.metrics.classification_report(..., output_dict=True)``.

    output_path : Path
        Filesystem path where the PNG should be written.

    Returns
    -------
    None
        The function writes the image to disk and does not return a
        value.

    Notes
    -----
    - The chart intentionally omits the colour scale because the
      numeric annotations carry the important information directly.
    """

    report_df = pd.DataFrame(classification_report_dict).transpose()
    plot_df = report_df.loc[
        [index for index in report_df.index if index != "accuracy"],
        ["precision", "recall", "f1-score", "support"],
    ].copy()

    # Format values as strings so support remains readable and integer-like.
    formatted_df = plot_df.copy()
    for column in ["precision", "recall", "f1-score"]:
        formatted_df[column] = formatted_df[column].map(lambda value: f"{value:.2f}")
    formatted_df["support"] = formatted_df["support"].map(lambda value: f"{int(value)}")

    fig_height = 1.6 + 0.55 * len(formatted_df.index)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.axis("off")
    ax.set_title("Classification Report", pad=16)

    table = ax.table(
        cellText=formatted_df.values,
        rowLabels=formatted_df.index,
        colLabels=formatted_df.columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.15, 1.8)

    # Apply simple styling for readability instead of a heatmap.
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B8C4D3")
        cell.set_linewidth(0.8)

        if row == 0:
            cell.set_facecolor("#DCEBFA")
            cell.set_text_props(weight="bold", color="#1F2937")
        elif col == -1:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(weight="bold", color="#111827")
        elif row % 2 == 0:
            cell.set_facecolor("#FAFAFA")
        else:
            cell.set_facecolor("#FFFFFF")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_multiclass_roc_curve(
    y_true,
    y_score,
    output_path: Path,
) -> dict[str, float]:
    """
    Render and save a multiclass one-vs-rest ROC chart.

    Parameters
    ----------
    y_true :
        True class labels encoded as integer targets ``0..2``.

    y_score :
        Per-class probability scores aligned with the class order
        used during training.

    output_path : Path
        Filesystem path where the PNG should be written.

    Returns
    -------
    dict[str, float]
        Dictionary mapping per-class AUC metric names to their scores.

    Notes
    -----
    - The plot contains three ROC curves:

        - setosa vs the other classes
        - versicolor vs the other classes
        - virginica vs the other classes
    """

    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(7, 5))
    roc_auc_scores: dict[str, float] = {}

    for idx, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
        roc_auc = auc(fpr, tpr)
        roc_auc_scores[f"auc_{class_name}"] = roc_auc
        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return roc_auc_scores


def save_learning_curve_plot(
    estimator,
    X,
    y,
    output_path: Path,
) -> None:
    """
    Render and save a learning-curve plot.

    Parameters
    ----------
    estimator :
        Fitted or unfitted scikit-learn compatible estimator.

    X :
        Feature matrix used to generate the learning-curve scores.

    y :
        Target labels aligned with ``X``.

    output_path : Path
        Filesystem path where the PNG should be written.

    Returns
    -------
    None
        The function writes the image to disk and does not return a
        value.
    """

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=5,
        scoring="accuracy",
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
    ax.plot(train_sizes, validation_mean, marker="o", label="Validation Accuracy")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_oob_error_curve(
    X,
    y,
    output_path: Path,
    best_params: dict,
    random_state: int,
) -> None:
    """
    Render and save an out-of-bag error curve for the best model.

    Parameters
    ----------
    X :
        Training feature matrix.

    y :
        Training target labels.

    output_path : Path
        Filesystem path where the PNG should be written.

    best_params : dict
        Best hyperparameters returned by GridSearchCV for the
        Random Forest model.

    random_state : int
        Seed used to keep the warm-start fitting sequence reproducible.

    Returns
    -------
    None
        The function writes the image to disk and does not return a
        value.

    Notes
    -----
    - This plot is a better Random Forest training-progress analogue
      than a traditional loss-vs-iteration curve.
    """

    target_n_estimators = int(best_params["n_estimators"])
    step = max(5, target_n_estimators // 10)
    estimator_steps = list(range(step, target_n_estimators + 1, step))

    if estimator_steps[-1] != target_n_estimators:
        estimator_steps.append(target_n_estimators)

    forest = RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        oob_score=True,
        bootstrap=True,
        random_state=random_state,
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
    )

    oob_errors = []

    for n_estimators in estimator_steps:
        forest.set_params(n_estimators=n_estimators)
        forest.fit(X, y)
        oob_errors.append(1 - forest.oob_score_)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(estimator_steps, oob_errors, marker="o")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("OOB Error")
    ax.set_title("Out-of-Bag Error Curve")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
