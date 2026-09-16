"""
tests/test_pipeline.py

Tests the SAVED PIPELINE directly -- independent of the FastAPI layer. These tests would
catch a bug in the model/feature-engineering logic even if the API layer were completely
broken, and vice versa (see test_api.py) -- keeping the two test files separate mirrors
keeping the two concerns (ML pipeline vs. web layer) separate in the actual project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import pytest

from features import FeatureCreator  # noqa: F401 -- required for joblib to unpickle the pipeline

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "exam_score_pipeline.joblib"


@pytest.fixture(scope="module")
def pipeline():
    if not MODEL_PATH.exists():
        pytest.skip(f"No trained model found at {MODEL_PATH} -- run the notebook first.")
    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_student():
    return pd.DataFrame([{
        "study_hours": 12.5,
        "attendance_pct": 88.0,
        "mock_test_1": 72.0,
        "mock_test_2": 75.0,
        "mock_test_3": 70.0,
        "income_bracket": "Medium",
        "city": "Pune",
        "enrollment_date": "2025-06-01",
        "shoe_size": 9.0,
        "lucky_number": 42,
    }])


def test_pipeline_loads_successfully(pipeline):
    assert pipeline is not None


def test_pipeline_predicts_a_reasonable_score(pipeline, sample_student):
    prediction = pipeline.predict(sample_student)[0]
    # exam scores in this dataset are 0-100 -- a real prediction should land in a sane range
    assert 0 <= prediction <= 110  # small headroom since Linear Regression can slightly overshoot


def test_pipeline_handles_missing_study_hours(pipeline, sample_student):
    """The pipeline was specifically trained to impute missing study_hours -- confirm it
    doesn't crash and still produces a sensible prediction when that value is missing."""
    student_missing_hours = sample_student.copy()
    student_missing_hours["study_hours"] = None
    prediction = pipeline.predict(student_missing_hours)[0]
    assert 0 <= prediction <= 110


def test_pipeline_handles_unseen_city(pipeline, sample_student):
    """A city never seen during training should NOT crash the pipeline -- the one-hot
    encoder was built with handle_unknown='ignore' specifically for this."""
    student_new_city = sample_student.copy()
    student_new_city["city"] = "Chennai"  # not in the training data's city list
    prediction = pipeline.predict(student_new_city)[0]
    assert 0 <= prediction <= 110


def test_higher_study_hours_predicts_higher_score(pipeline, sample_student):
    """A basic sanity/directionality check: all else equal, more study hours should not
    predict a LOWER score, given the model's coefficient for study_hours is positive."""
    low_effort = sample_student.copy()
    low_effort["study_hours"] = 2.0

    high_effort = sample_student.copy()
    high_effort["study_hours"] = 20.0

    pred_low = pipeline.predict(low_effort)[0]
    pred_high = pipeline.predict(high_effort)[0]
    assert pred_high > pred_low


def test_pipeline_output_is_deterministic(pipeline, sample_student):
    """Calling predict() twice on the same input must give the same result -- a basic
    reproducibility check."""
    pred_1 = pipeline.predict(sample_student)[0]
    pred_2 = pipeline.predict(sample_student)[0]
    assert pred_1 == pred_2


def test_irrelevant_columns_do_not_change_prediction_much(pipeline, sample_student):
    """shoe_size and lucky_number were deliberately dropped by feature selection --
    changing them should NOT meaningfully change the prediction."""
    baseline_pred = pipeline.predict(sample_student)[0]

    changed = sample_student.copy()
    changed["shoe_size"] = 20.0
    changed["lucky_number"] = 999
    changed_pred = pipeline.predict(changed)[0]

    assert abs(baseline_pred - changed_pred) < 0.01
