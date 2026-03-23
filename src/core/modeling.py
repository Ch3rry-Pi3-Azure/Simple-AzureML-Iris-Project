"""
Shared model-building and hyperparameter search utilities.

This module provides a small abstraction layer over the project's
Random Forest modelling strategy. It exists so that both the local
training workflow and the Azure ML pipeline training step can use
exactly the same estimator configuration and grid-search logic.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# Default hyperparameter search space used by local and pipeline training.
#   - The grid is intentionally modest because the Iris dataset is small.
#   - This keeps the example educational and fast to run.
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 5, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}


def get_base_model(random_state: int = 5901) -> RandomForestClassifier:
    """
    Construct the base Random Forest classifier used in this project.

    Parameters
    ----------
    random_state : int, default=5901
        Seed used to make the estimator behaviour reproducible.

    Returns
    -------
    RandomForestClassifier
        Unfitted scikit-learn Random Forest estimator configured
        with the supplied random state.
    """

    return RandomForestClassifier(random_state=random_state)


def run_grid_search(
    X_train,
    y_train,
    random_state: int = 5901,
    scoring: str = "accuracy",
    cv: int = 5,
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Run a GridSearchCV sweep for the project's Random Forest model.

    Parameters
    ----------
    X_train :
        Training feature matrix.

    y_train :
        Training target labels.

    random_state : int, default=5901
        Seed applied to the base estimator before the hyperparameter
        search begins.

    scoring : str, default="accuracy"
        Metric used by scikit-learn to rank parameter combinations.

    cv : int, default=5
        Number of cross-validation folds used during grid search.

    n_jobs : int, default=-1
        Number of parallel workers used by scikit-learn. A value of
        ``-1`` instructs scikit-learn to use all available cores.

    Returns
    -------
    GridSearchCV
        Fitted scikit-learn grid-search object containing the best
        estimator, best score, best parameters, and full CV results.

    Notes
    -----
    - The returned search object is already fitted and ready for use.

    - The best estimator can be accessed through ``best_estimator_``.

    - Detailed per-combination results are available via
      ``cv_results_``.
    """

    search = GridSearchCV(
        estimator=get_base_model(random_state=random_state),
        param_grid=DEFAULT_PARAM_GRID,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )

    search.fit(X_train, y_train)
    return search
