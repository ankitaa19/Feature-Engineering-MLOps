"""
src/evaluation/model_selection.py

WHY?
    Cross-validation (see `cross_validation.py`) answers "how good is THIS ONE model
    configuration, honestly?" GridSearchCV answers a DIFFERENT question: "which
    configuration, out of many I'm considering, is best?" It does that by running
    cross-validation ONCE PER hyperparameter combination in your grid, then picking the
    winner. In other words: cross-validation is the evaluation tool: GridSearchCV is
    cross-validation applied repeatedly, in a search, to choose hyperparameters. If you
    only cross-validate, you get an honest score for one configuration; if you only
    grid-search, you still need a genuinely held-out test set afterward to report an
    unbiased final number, because the grid search process itself has "seen" the
    cross-validation folds many times while picking a winner.

    `compare_models()` exists because a common next question is "which of these several
    MODEL TYPES should I even be tuning?" -- a fast, consistent way to shortlist before
    you spend time tuning any one of them in depth.

WHAT?
    - `tune_model()`: wraps `GridSearchCV` -- exhaustively tries every combination in a
      parameter grid.
    - `random_search_model()`: wraps `RandomizedSearchCV` -- samples a fixed number of
      random combinations, much cheaper when the grid is large.
    - `compare_models()`: fits several named models (from `get_model()`) on the same
      preprocessed data, cross-validates each, and returns a ranked DataFrame.

HOW?
    from src.evaluation.model_selection import tune_model, random_search_model, compare_models

    tuned = tune_model(pipeline, param_grid={"model__alpha": [0.1, 1, 10]}, X=X_train, y=y_train, cv=5)
    print(tuned["best_params_"], tuned["best_score_"])
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.models.factory import get_model
from src.evaluation.cross_validation import cross_validate_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def tune_model(pipeline, param_grid: dict, X, y, cv: int = 5, scoring: str | None = None) -> dict:
    """Exhaustively search a parameter grid using GridSearchCV.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        An unfit pipeline. Parameter grid keys must use the `stepname__paramname` format,
        e.g. `"model__alpha"` to tune the `alpha` of the pipeline's "model" step.
    param_grid : dict
        e.g. {"model__alpha": [0.01, 0.1, 1, 10], "model__max_iter": [1000, 5000]}
    X, y : array-like
        Training data.
    cv : int, default 5
        Number of cross-validation folds evaluated per parameter combination.
    scoring : str, optional
        Scikit-learn scoring string. Defaults to the estimator's own `.score()` if not given.

    Returns
    -------
    dict with keys: "best_estimator_", "best_params_", "best_score_", "cv_results_" (as a
    DataFrame, sorted best-first).
    """
    search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    search.fit(X, y)

    logger.info(f"GridSearchCV best score: {search.best_score_:.4f}, best params: {search.best_params_}")

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return {
        "best_estimator_": search.best_estimator_,
        "best_params_": search.best_params_,
        "best_score_": search.best_score_,
        "cv_results_": cv_results_df,
    }


def random_search_model(
    pipeline, param_distributions: dict, X, y, cv: int = 5, n_iter: int = 20,
    scoring: str | None = None, random_state: int = 42,
) -> dict:
    """Search a parameter space using RandomizedSearchCV -- samples `n_iter` random
    combinations instead of trying every single one. Much cheaper than `tune_model()`
    when the grid is large or continuous.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        An unfit pipeline.
    param_distributions : dict
        Same key format as `tune_model`'s `param_grid`; values can be lists OR
        scipy.stats distributions for continuous sampling.
    X, y : array-like
        Training data.
    cv : int, default 5
        Cross-validation folds per sampled combination.
    n_iter : int, default 20
        Number of random combinations to try.
    scoring : str, optional
        Scikit-learn scoring string.
    random_state : int, default 42
        For reproducible sampling.

    Returns
    -------
    dict with the same keys as `tune_model()`.
    """
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_distributions, cv=cv, n_iter=n_iter,
        scoring=scoring, random_state=random_state, n_jobs=-1,
    )
    search.fit(X, y)

    logger.info(f"RandomizedSearchCV best score: {search.best_score_:.4f}, best params: {search.best_params_}")

    cv_results_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return {
        "best_estimator_": search.best_estimator_,
        "best_params_": search.best_params_,
        "best_score_": search.best_score_,
        "cv_results_": cv_results_df,
    }


def compare_models(
    model_names: list[str],
    preprocessor,
    X_train, y_train, X_test, y_test,
    cv: int = 5,
    scoring: str = "accuracy",
    stratified: bool = True,
) -> pd.DataFrame:
    """Fit, cross-validate, and test-evaluate several models on the same data, for a quick
    shortlist before deciding which one to tune further.

    Parameters
    ----------
    model_names : list[str]
        Names registered in `get_model()`, e.g. ["ridge", "lasso", "elasticnet"].
    preprocessor : sklearn ColumnTransformer
        Applied identically to every model, so the comparison isolates the MODEL choice.
    X_train, y_train, X_test, y_test : array-like
        Train/test split. Cross-validation uses the training split only.
    cv : int, default 5
        Cross-validation folds.
    scoring : str, default "accuracy"
        Scoring metric for BOTH cross-validation and the reported test score. Use a
        regression-appropriate metric (e.g. "r2") for regression models.
    stratified : bool, default True
        Passed through to `cross_validate_model` -- set False for regression.

    Returns
    -------
    pd.DataFrame with columns: "Model", "CV Mean", "CV Std", "Test Score" -- sorted by
    CV Mean, best first.
    """
    from src.preprocessing.pipeline import build_pipeline
    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring)
    rows = []

    for name in model_names:
        model = get_model(name)
        pipeline = build_pipeline(preprocessor, model)

        cv_result = cross_validate_model(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, stratified=stratified
        )

        pipeline.fit(X_train, y_train)
        test_score = scorer(pipeline, X_test, y_test)

        rows.append({
            "Model": name,
            "CV Mean": cv_result["mean"],
            "CV Std": cv_result["std"],
            "Test Score": test_score,
        })
        logger.info(f"compare_models -- {name}: CV={cv_result['mean']:.4f}, Test={test_score:.4f}")

    results_df = pd.DataFrame(rows).sort_values("CV Mean", ascending=False).reset_index(drop=True)
    return results_df
