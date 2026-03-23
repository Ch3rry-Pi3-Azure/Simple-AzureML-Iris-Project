"""
Shared model-building and hyperparameter search utilities.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 5, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}


def get_base_model(random_state: int = 5901) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=random_state)


def run_grid_search(
    X_train,
    y_train,
    random_state: int = 5901,
    scoring: str = "accuracy",
    cv: int = 5,
    n_jobs: int = -1,
) -> GridSearchCV:
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
