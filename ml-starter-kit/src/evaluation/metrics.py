"""
src/evaluation/metrics.py

WHY?
    "How good is this model?" always means computing the same handful of metrics --
    but which metrics, and how you call them, differs between regression and
    classification. Wrapping both in one place with a consistent dict/DataFrame return
    shape means every project reports results the same way, which makes them
    comparable across experiments.

WHAT?
    - `evaluate_regression(y_true, y_pred)`: MAE, MSE, RMSE, R².
    - `evaluate_classification(y_true, y_pred, y_proba=None)`: Accuracy, Precision,
      Recall, F1, and ROC-AUC (only if `y_proba` is provided -- AUC needs probabilities,
      not just hard predictions).
    - `evaluate_model(pipeline, X_test, y_test, task=None)`: the one-call convenience
      version -- runs `.predict()` (and `.predict_proba()` if available) for you and
      routes to the right metric function.

HOW?
    from src.evaluation.metrics import evaluate_model
    results = evaluate_model(pipeline, X_test, y_test, task="classification")
"""
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_regression(y_true, y_pred) -> dict:
    """Compute standard regression metrics.

    Returns
    -------
    dict with keys: "MAE", "MSE", "RMSE", "R2"
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_classification(y_true, y_pred, y_proba=None, average: str = "binary") -> dict:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
        True labels and hard (class) predictions.
    y_proba : array-like, optional
        Predicted probabilities for the positive class (needed for ROC-AUC). If not
        provided, "ROC_AUC" is omitted from the result rather than silently wrong.
    average : str, default "binary"
        Averaging strategy for Precision/Recall/F1 -- use "binary" for 2-class problems,
        "macro" or "weighted" for multi-class.

    Returns
    -------
    dict with keys: "Accuracy", "Precision", "Recall", "F1", and "ROC_AUC" if `y_proba`
    was given.
    """
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_proba is not None:
        try:
            results["ROC_AUC"] = roc_auc_score(y_true, y_proba)
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
    return results


def evaluate_model(pipeline, X_test, y_test, task: str | None = None) -> dict:
    """One-call evaluation: predicts with `pipeline` and routes to the right metric set.

    Parameters
    ----------
    pipeline : fitted sklearn estimator/Pipeline
        Must already be fit.
    X_test, y_test : array-like
        Held-out test data.
    task : {"regression", "classification"}, optional
        If not given, inferred from whether `pipeline` has `predict_proba` (classification)
        or not (regression) -- pass it explicitly if you want to be certain.

    Returns
    -------
    dict of metric name -> value
    """
    if task is None:
        task = "classification" if hasattr(pipeline, "predict_proba") else "regression"

    y_pred = pipeline.predict(X_test)

    if task == "regression":
        results = evaluate_regression(y_test, y_pred)
    elif task == "classification":
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)
            # take the positive-class column for binary classification
            y_proba = proba[:, 1] if proba.shape[1] == 2 else None
        results = evaluate_classification(y_test, y_pred, y_proba=y_proba)
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'.")

    logger.info(f"Evaluation ({task}): {results}")
    return results


def results_to_dataframe(results: dict, model_name: str = "model") -> pd.DataFrame:
    """Turn a metrics dict into a one-row DataFrame, handy for stacking multiple models'
    results together with pd.concat()."""
    return pd.DataFrame([results], index=[model_name])
