"""
Utilities for loading and splitting the Iris dataset.

This module provides a small helper function used in simple
classification demonstrations. It loads the Iris dataset from
scikit-learn and prepares a reproducible train–test split.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data(test_size: float = 0.2, random_state: int = 5901):
    """
    Load the Iris dataset and produce a stratified train–test split.

    The function retrieves the Iris dataset using scikit-learn's
    built-in loader and separates the features and labels. The
    dataset is then split into training and testing subsets using
    a stratified sampling strategy so that class proportions are
    preserved in both partitions.

    Parameters
    ----------
    test_size : float
        Fraction of the dataset reserved for the test set.

        For example, a value of ``0.2`` means that 20% of the
        observations will be placed in the test partition while
        the remaining 80% are used for training.

    random_state : int
        Random seed used by the train–test split.

        Providing a fixed seed ensures reproducibility so that
        repeated runs produce identical dataset partitions.

    Returns
    -------
    tuple
        Four objects returned in the following order:

        X_train : pandas.DataFrame
            Feature matrix used to train the model.

        X_test : pandas.DataFrame
            Feature matrix used for evaluation.

        y_train : pandas.Series
            Target labels corresponding to ``X_train``.

        y_test : pandas.Series
            Target labels corresponding to ``X_test``.

    Notes
    -----
    - The Iris dataset contains **150 observations** of iris flowers
      across three species:

        1. *Setosa*
        2. *Versicolor*
        3. *Virginica*

    - Each observation includes four numerical features:

        - sepal length
        - sepal width
        - petal length
        - petal width

    - Stratified splitting is used to ensure that the class
      distribution remains consistent between training and test sets.

    Example
    -------
    Load the dataset and inspect the resulting shapes.

    >>> X_train, X_test, y_train, y_test = load_data()

    >>> X_train.shape
    (120, 4)

    >>> X_test.shape
    (30, 4)
    """

    # Load Iris dataset as a pandas DataFrame
    #   - `as_frame=True` returns feature data as a DataFrame
    #     and labels as a pandas Series
    iris = load_iris(as_frame=True)

    # Separate features and target labels
    X = iris.data
    y = iris.target

    # Perform a train–test split
    #   - Stratification preserves class balance
    #   - Random state ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Return partitioned datasets
    return X_train, X_test, y_train, y_test