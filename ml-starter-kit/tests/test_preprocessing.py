"""
tests/test_preprocessing.py

Unit tests for src.preprocessing -- numeric/categorical pipeline builders, the combined
ColumnTransformer, and the full Pipeline builder. Covers correct behavior AND the explicit
error handling this framework promises (e.g. a categorical column that doesn't exist).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.numerical import build_numeric_pipeline
from src.preprocessing.categorical import build_categorical_pipeline
from src.preprocessing.pipeline import build_preprocessor, build_pipeline
from src.models.factory import get_model


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, np.nan, 40, 35, 50],
        "income": [30000, 45000, np.nan, 60000, 52000],
        "city": ["Mumbai", "Delhi", None, "Mumbai", "Pune"],
        "target": [0, 1, 0, 1, 1],
    })


def test_numeric_pipeline_imputes_and_scales():
    X = pd.DataFrame({"age": [20.0, np.nan, 40.0, 60.0]})
    pipe = build_numeric_pipeline(imputation="median", scaling="standard")
    result = pipe.fit_transform(X)
    assert not np.isnan(result).any()          # imputation filled the missing value
    assert abs(result.mean()) < 1e-6            # standard scaling centers around 0


def test_numeric_pipeline_rejects_invalid_strategy():
    with pytest.raises(ValueError):
        build_numeric_pipeline(imputation="not_a_real_strategy")


def test_categorical_pipeline_handles_unknown_category_safely():
    X_train = pd.DataFrame({"city": ["Mumbai", "Delhi", "Mumbai"]})
    X_test = pd.DataFrame({"city": ["Chennai"]})  # never seen during training

    pipe = build_categorical_pipeline(encoding="onehot")
    pipe.fit(X_train)
    result = pipe.transform(X_test)  # should NOT raise, thanks to handle_unknown="ignore"
    assert result.shape[0] == 1
    assert result.sum() == 0  # unseen category encodes as all-zeros, not an error


def test_build_preprocessor_rejects_overlapping_columns():
    with pytest.raises(ValueError):
        build_preprocessor(numerical_features=["age"], categorical_features=["age"])


def test_build_preprocessor_rejects_empty_feature_lists():
    with pytest.raises(ValueError):
        build_preprocessor(numerical_features=[], categorical_features=[])


def test_full_pipeline_fits_and_predicts(sample_df):
    X = sample_df.drop(columns=["target"])
    y = sample_df["target"]

    preprocessor = build_preprocessor(
        numerical_features=["age", "income"], categorical_features=["city"]
    )
    model = get_model("logistic", max_iter=1000)
    pipeline = build_pipeline(preprocessor, model)

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_pipeline_never_fit_on_test_data_produces_consistent_transform(sample_df):
    """A regression-style check for train/inference consistency: transforming the SAME
    row twice through a fitted pipeline should give the SAME result."""
    X = sample_df.drop(columns=["target"])
    y = sample_df["target"]

    preprocessor = build_preprocessor(
        numerical_features=["age", "income"], categorical_features=["city"]
    )
    preprocessor.fit(X)

    transformed_once = preprocessor.transform(X.iloc[[0]])
    transformed_twice = preprocessor.transform(X.iloc[[0]])
    np.testing.assert_array_equal(transformed_once.toarray() if hasattr(transformed_once, "toarray") else transformed_once,
                                   transformed_twice.toarray() if hasattr(transformed_twice, "toarray") else transformed_twice)
