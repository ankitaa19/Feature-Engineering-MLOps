"""
src/feature_engineering/selection.py

WHY?
    Not every feature you engineer earns its place. Filter methods (fast, model-agnostic),
    wrapper methods (slower, but tuned to your actual model), and embedded methods (built
    into training itself) each answer "which features matter?" a different way -- having
    all three one import away makes it easy to check whether they agree.

WHAT?
    - `select_k_best()`: filter method, ranks features by a statistical test.
    - `rfe_selector()`: wrapper method, iteratively removes the weakest feature.
    - `select_from_model()`: embedded method, keeps whatever a fitted model already
      considers important (e.g. non-zero Lasso coefficients, tree importances).

HOW?
    from src.feature_engineering.selection import select_k_best, rfe_selector, select_from_model
    selector = select_k_best(task="classification", k=10)

IMPORTANT -- chi-square and negative values
    Chi-square (`score_func="chi2"`) is only defined for NON-NEGATIVE features (it's a test
    built on counts/frequencies). If your data can be negative (most standardized/scaled
    data will be), `select_k_best(score_func="chi2", ...)` will raise a clear error rather
    than silently producing nonsense -- scale AFTER selection, or use "f_classif" /
    "mutual_info_classif" instead, if your features are already standardized.
"""
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel,
)
from sklearn.linear_model import Lasso

_CLASSIFICATION_SCORERS = {"chi2": chi2, "f_classif": f_classif, "mutual_info_classif": mutual_info_classif}
_REGRESSION_SCORERS = {"f_regression": f_regression, "mutual_info_regression": mutual_info_regression}


def select_k_best(task: str = "classification", score_func: str = "f_classif", k: int = 10) -> SelectKBest:
    """Build a filter-based feature selector (SelectKBest).

    Parameters
    ----------
    task : {"classification", "regression"}, default "classification"
        Determines which score functions are valid.
    score_func : str, default "f_classif"
        Classification: "chi2", "f_classif", or "mutual_info_classif".
        Regression: "f_regression" or "mutual_info_regression".
    k : int, default 10
        Number of top features to keep.

    Returns
    -------
    sklearn.feature_selection.SelectKBest

    Raises
    ------
    ValueError
        If `score_func` doesn't match `task`, or isn't recognized.
    """
    if task == "classification":
        if score_func not in _CLASSIFICATION_SCORERS:
            raise ValueError(f"For task='classification', score_func must be one of {list(_CLASSIFICATION_SCORERS)}.")
        func = _CLASSIFICATION_SCORERS[score_func]
    elif task == "regression":
        if score_func not in _REGRESSION_SCORERS:
            raise ValueError(f"For task='regression', score_func must be one of {list(_REGRESSION_SCORERS)}.")
        func = _REGRESSION_SCORERS[score_func]
    else:
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'.")

    return SelectKBest(score_func=func, k=k)


def check_nonnegative_for_chi2(X) -> None:
    """Raise a clear error if X contains negative values -- call this before fitting
    a chi2-based selector on data that might have been scaled/standardized already."""
    X_arr = np.asarray(X)
    if (X_arr < 0).any():
        raise ValueError(
            "chi2 requires all feature values to be non-negative (it's a test built on "
            "counts/frequencies). Your data contains negative values -- this usually means "
            "it's already been standardized/scaled. Either select features BEFORE scaling, "
            "or use score_func='f_classif' / 'mutual_info_classif' instead."
        )


def rfe_selector(estimator, n_features_to_select: int | float = 0.5) -> RFE:
    """Build a wrapper-based feature selector (Recursive Feature Elimination).

    Parameters
    ----------
    estimator : sklearn estimator
        Must expose `.coef_` or `.feature_importances_` after fitting (e.g. LogisticRegression,
        LinearRegression, RandomForestClassifier). Typically from `get_model()`.
    n_features_to_select : int or float, default 0.5
        Number (int) or fraction (float) of features to keep.

    Returns
    -------
    sklearn.feature_selection.RFE

    Note
    ----
    RFE retrains the model once per feature removed -- much slower than a filter method.
    It's also MODEL-DEPENDENT: different estimators can produce different rankings on the
    same data (see Unit 3, Wrapper Methods).
    """
    return RFE(estimator=estimator, n_features_to_select=n_features_to_select)


def select_from_model(estimator, threshold: str | float = "median") -> SelectFromModel:
    """Build an embedded feature selector that keeps whatever the fitted model already
    considers important.

    Parameters
    ----------
    estimator : sklearn estimator
        Must expose `.coef_` or `.feature_importances_` after fitting. A Lasso/LassoCV
        model naturally zeroes out weak features; a tree-based model ranks by impurity
        reduction.
    threshold : str or float, default "median"
        Features scoring at or above this threshold are kept. Accepts sklearn's string
        shortcuts ("median", "mean", "1.25*mean") or a specific float.

    Returns
    -------
    sklearn.feature_selection.SelectFromModel
    """
    return SelectFromModel(estimator=estimator, threshold=threshold)


def lasso_selector(alpha: float = 0.01, threshold: str | float = 1e-5) -> SelectFromModel:
    """Convenience wrapper: embedded feature selection using Lasso's own coefficients.

    Equivalent to `select_from_model(Lasso(alpha=alpha), threshold=threshold)` -- a
    dedicated shortcut since Lasso-based selection is common enough to deserve one.

    Parameters
    ----------
    alpha : float, default 0.01
        Lasso's regularization strength. Higher alpha -> fewer features survive.
    threshold : str or float, default 1e-5
        Coefficients with absolute value below this are treated as "zeroed out."

    Returns
    -------
    sklearn.feature_selection.SelectFromModel
    """
    return SelectFromModel(estimator=Lasso(alpha=alpha), threshold=threshold)
