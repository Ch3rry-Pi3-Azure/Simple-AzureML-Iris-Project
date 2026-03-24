"""
Reusable preprocessing pipeline helpers for model training.

This module keeps the scikit-learn preprocessing assembly separate
from the model-selection logic so the repository has a clearer
template for future projects. The current preprocessing flow is
deliberately simple:

1. build the shared reusable derived features
2. optionally apply numeric scaling

Even though the current Random Forest model does not require scaling,
keeping that option in a dedicated preprocessing module makes it much
easier to reuse the repository structure for models that do.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    from .features import build_model_feature_frame
except ImportError:
    from features import build_model_feature_frame


def _build_model_feature_frame_for_pipeline(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Adapter used by scikit-learn's ``FunctionTransformer``.

    Parameters
    ----------
    dataset : pd.DataFrame
        Raw Iris measurement frame passed into the training pipeline.

    Returns
    -------
    pd.DataFrame
        Full model feature matrix containing raw and derived columns.
    """

    return build_model_feature_frame(dataset)


def build_preprocessing_pipeline(use_scaling: bool = False) -> Pipeline:
    """
    Build the preprocessing pipeline shared by training workflows.

    Parameters
    ----------
    use_scaling : bool, default=False
        Whether a ``StandardScaler`` step should be applied after the
        reusable derived features are generated.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing feature generation and an
        optional scaling step.

    Notes
    -----
    - When ``use_scaling`` is ``False``, the scaling step is replaced
      with ``"passthrough"`` so the pipeline shape remains stable.
    """

    return Pipeline(
        steps=[
            (
                "feature_builder",
                FunctionTransformer(
                    _build_model_feature_frame_for_pipeline,
                    validate=False,
                ),
            ),
            (
                "scaler",
                StandardScaler() if use_scaling else "passthrough",
            ),
        ]
    )
