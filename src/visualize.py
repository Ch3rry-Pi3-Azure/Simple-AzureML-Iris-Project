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
from sklearn.ensemble import RandomForestClassifier
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
    ax.imshow(plot_df.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
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


def save_oob_error_curve(
    X,
    y,
    output_path: Path,
    best_params: dict,
    random_state: int,
) -> None:
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
