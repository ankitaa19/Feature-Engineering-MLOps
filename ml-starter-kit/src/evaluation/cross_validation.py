"""
src/evaluation/cross_validation.py

WHY?
    A single train/test split gives you one number -- and one number can be lucky or
    unlucky. Cross-validation fits and scores your pipeline on several different splits
    and reports the spread, giving you a much more honest sense of how stable your
    model's performance actually is.

WHAT?
    `cross_validate_model()` wraps scikit-learn's `cross_val_score`, automatically choosing
    `StratifiedKFold` for classification (to keep class proportions consistent across folds)
    or plain `KFold` for regression, and returns individual fold scores plus their mean/std.

HOW?
    from src.evaluation.cross_validation import cross_validate_model
    results = cross_validate_model(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(results["mean"], results["std"])
"""
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

from src.utils.logger import get_logger

logger = get_logger(__name__)


def cross_validate_model(
    pipeline,
    X,
    y,
    cv: int = 5,
    scoring: str = "accuracy",
    stratified: bool = True,
    random_state: int = 42,
) -> dict:
    """Cross-validate a pipeline and return fold scores plus summary statistics.

    Parameters
    ----------
    pipeline : sklearn estimator/Pipeline
        An UNFIT pipeline -- cross-validation fits a fresh copy on each fold internally.
    X, y : array-like
        Training data (use the TRAINING split only -- never include test data here).
    cv : int, default 5
        Number of folds.
    scoring : str, default "accuracy"
        Any scikit-learn scoring string (e.g. "accuracy", "roc_auc", "r2",
        "neg_mean_squared_error").
    stratified : bool, default True
        If True, use StratifiedKFold (keeps class proportions consistent per fold) --
        appropriate for classification. Set False for regression, where stratification
        on a continuous target doesn't apply.
    random_state : int, default 42
        Seed for the fold shuffling, for reproducibility.

    Returns
    -------
    dict with keys:
        "scores" : np.ndarray of per-fold scores
        "mean"   : float, mean across folds
        "std"    : float, standard deviation across folds
    """
    if stratified:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    scores = cross_val_score(pipeline, X, y, cv=splitter, scoring=scoring)

    result = {"scores": scores, "mean": scores.mean(), "std": scores.std()}
    logger.info(
        f"Cross-validation ({cv}-fold, scoring={scoring}): "
        f"mean={result['mean']:.4f}, std={result['std']:.4f}"
    )
    return result
