"""
tests/test_models.py

Unit tests for src.models.factory -- covering successful model creation across both
registries, hyperparameter pass-through, the explicit error for unknown model names,
and the register_model() extension point.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.factory import get_model, list_available_models, register_model


def test_get_model_returns_correct_class_for_regression():
    model = get_model("ridge", alpha=0.5)
    assert isinstance(model, Ridge)
    assert model.alpha == 0.5


def test_get_model_returns_correct_class_for_classification():
    model = get_model("logistic", C=2.0)
    assert isinstance(model, LogisticRegression)
    assert model.C == 2.0


def test_get_model_passes_through_multiple_hyperparameters():
    model = get_model("random_forest_classifier", n_estimators=50, max_depth=5, random_state=1)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 50
    assert model.max_depth == 5
    assert model.random_state == 1


def test_get_model_unknown_name_raises_clear_error():
    with pytest.raises(ValueError) as exc_info:
        get_model("not_a_real_model")
    # the error message should actually help -- list what IS available
    assert "Unknown model" in str(exc_info.value)
    assert "Available regression models" in str(exc_info.value)


def test_list_available_models_returns_both_tasks():
    available = list_available_models()
    assert "regression" in available
    assert "classification" in available
    assert "ridge" in available["regression"]
    assert "logistic" in available["classification"]


def test_register_model_makes_new_model_available():
    from sklearn.linear_model import BayesianRidge

    register_model("bayesian_ridge", BayesianRidge, task="regression")
    model = get_model("bayesian_ridge")
    assert isinstance(model, BayesianRidge)
    assert "bayesian_ridge" in list_available_models()["regression"]


def test_get_model_returns_a_fresh_instance_each_time():
    """Two calls to get_model() must not return the SAME object -- otherwise fitting
    one would silently mutate the other."""
    model_a = get_model("ridge")
    model_b = get_model("ridge")
    assert model_a is not model_b
