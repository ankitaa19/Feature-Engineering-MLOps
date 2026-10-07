"""
app/main.py

A minimal FastAPI application that loads the pipeline saved by the notebook
(models/exam_score_pipeline.joblib) and serves predictions over HTTP.

This is the "notebook -> real application" step: everything the pipeline learned during
training (imputation medians, scaler means, one-hot categories, PCA directions, which
features survived selection, and the model's coefficients) is already baked into that ONE
file. This app's only job is to load it once, validate incoming requests, and call
.predict() -- it contains ZERO feature engineering logic of its own, on purpose. That is
exactly what prevents "the API preprocesses data slightly differently than training did,"
one of the most common real causes of a model behaving worse in production than in testing.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Then try:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
        "study_hours": 12.5, "attendance_pct": 88.0,
        "mock_test_1": 72.0, "mock_test_2": 75.0, "mock_test_3": 70.0,
        "income_bracket": "Medium", "city": "Pune",
        "enrollment_date": "2025-06-01", "shoe_size": 9.0, "lucky_number": 42
    }'
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# IMPORTANT: this import is not decorative. The pickled pipeline contains a custom
# FeatureCreator step, and joblib only stores a REFERENCE to where that class is defined,
# not its code. Without this import, loading the pickle below fails with
# "AttributeError: Can't get attribute 'FeatureCreator'" -- see features.py's docstring
# and the notebook's explanation for the full story of this exact bug.
from features import FeatureCreator  # noqa: F401

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "exam_score_pipeline.joblib"
)

# Loaded once, at startup -- NOT on every request. This is the same fitted pipeline object
# built and verified in notebooks/exam_score_pipeline.ipynb, nothing re-implemented here.
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run the notebook first to train and "
            f"save the pipeline."
        )
    pipeline = joblib.load(MODEL_PATH)
    yield
    pipeline = None  # nothing else to clean up -- included for symmetry/clarity


app = FastAPI(
    title="Student Exam Score Predictor",
    description="Predicts a student's final exam score from raw, unprocessed profile data. "
    "All feature engineering (imputation, scaling, encoding, PCA, feature "
    "selection) happens automatically inside the loaded pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


class StudentInput(BaseModel):
    """Raw, unprocessed student data -- exactly the same raw columns the pipeline was
    trained on. No preprocessed/derived fields belong here; the pipeline computes those
    itself."""

    study_hours: float = Field(..., ge=0, le=40, description="Weekly study hours")
    attendance_pct: float = Field(
        ..., ge=0, le=100, description="Attendance percentage"
    )
    mock_test_1: float = Field(..., ge=0, le=100)
    mock_test_2: float = Field(..., ge=0, le=100)
    mock_test_3: float = Field(..., ge=0, le=100)
    income_bracket: Literal["Low", "Medium", "High"]
    city: str
    enrollment_date: str = Field(..., description="YYYY-MM-DD")
    shoe_size: float = Field(
        ...,
        description="Deliberately irrelevant -- kept to show the pipeline ignores it",
    )
    lucky_number: int = Field(
        ...,
        description="Deliberately irrelevant -- kept to show the pipeline ignores it",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
            }
        }
    )


class PredictionOutput(BaseModel):
    predicted_final_score: float


@app.get("/health")
def health():
    """Simple liveness check -- confirms the app is up AND the model actually loaded."""
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(student: StudentInput):
    """Predict a student's final exam score from raw input data.

    The pipeline handles imputation, scaling, encoding, PCA, and feature selection
    internally -- this endpoint just validates the request shape and calls .predict().
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_df = pd.DataFrame([student.model_dump()])

    try:
        prediction = pipeline.predict(input_df)[0]
    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {error}",
        ) from error

    return PredictionOutput(predicted_final_score=round(float(prediction), 1))
