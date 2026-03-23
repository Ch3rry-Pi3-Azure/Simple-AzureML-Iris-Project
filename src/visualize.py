"""
Visualization helpers for training and evaluation artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def save_confusion_matrix_plot(confusion_matrix, output_path: Path) -> None:
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
    report_df = pd.DataFrame(classification_report_dict).transpose()
    plot_df = report_df.loc[
        [index for index in report_df.index if index != "accuracy"],
        ["precision", "recall", "f1-score", "support"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    heatmap = ax.imshow(plot_df.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns)
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Classification Report")

    for row_idx in range(plot_df.shape[0]):
        for col_idx in range(plot_df.shape[1]):
            value = plot_df.iat[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_multiclass_roc_curve(
    y_true,
    y_score,
    output_path: Path,
) -> dict[str, float]:
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
