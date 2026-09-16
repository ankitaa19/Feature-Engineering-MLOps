"""
streamlit_app.py

A UI on top of the FastAPI prediction service -- NOT a second copy of the model.

WHY call the API instead of loading models/exam_score_pipeline.joblib directly in
Streamlit? Two reasons that matter once this is a real application, not just a demo:

  1. ONE SOURCE OF TRUTH. If Streamlit loaded the pipeline itself, you'd have TWO places
     that know how to make a prediction (the API and the UI) -- and the moment someone
     changes one without the other (a new model version, a bugfix), they silently
     disagree. Every prediction, from every client, should go through the same door.
  2. This is exactly how a real product is usually built: the API is the actual service
     (anyone can call it -- a mobile app, another team's backend, this UI); Streamlit is
     just ONE client of that service, with no special privileges.

So this file is deliberately "dumb": it collects input, calls the API over plain HTTP,
and displays whatever comes back. No feature engineering, no joblib.load(), nothing that
duplicates what app/main.py already does.

Run standalone (needs the API running separately -- see README):
    uvicorn app.main:app --port 8000          # terminal 1
    streamlit run streamlit_app.py            # terminal 2

Inside the combined Docker image, both processes run in the SAME container, so
API_URL defaults to http://localhost:8000 and just works without any configuration --
see docker/entrypoint.sh.
"""
import os
from datetime import date

import requests
import streamlit as st

# Configurable so the same code works: (a) locally, both processes on localhost, (b) in
# the combined container, both processes on localhost, and (c) in an unusual setup where
# someone points the UI at an API running somewhere else entirely.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Exam Score Predictor")
st.caption(
    "A UI for the FastAPI service at `%s` -- every prediction below goes through that "
    "same API, using the same trained pipeline. This page does no prediction logic of "
    "its own." % API_URL
)

# ---- sidebar: is the API actually reachable? -------------------------------------------
with st.sidebar:
    st.subheader("Service status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health.get("model_loaded"):
            st.success("API is up, model loaded")
        else:
            st.warning("API is up, but the model hasn't loaded yet")
        st.caption(f"model_version: {health.get('model_version', 'n/a')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Cannot reach API at {API_URL}")
        st.caption(str(e))

# ---- the input form ----------------------------------------------------------------------
st.subheader("Student details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input("Weekly study hours", min_value=0.0, max_value=40.0, value=12.5, step=0.5)
        attendance_pct = st.slider("Attendance %", min_value=0.0, max_value=100.0, value=88.0)
        mock_test_1 = st.number_input("Mock test 1 score", min_value=0.0, max_value=100.0, value=72.0)
        mock_test_2 = st.number_input("Mock test 2 score", min_value=0.0, max_value=100.0, value=75.0)
        mock_test_3 = st.number_input("Mock test 3 score", min_value=0.0, max_value=100.0, value=70.0)

    with col2:
        income_bracket = st.selectbox("Income bracket", ["Low", "Medium", "High"], index=1)
        city = st.selectbox("City", ["Mumbai", "Delhi", "Bengaluru", "Pune", "Other"], index=3)
        enrollment_date = st.date_input("Enrollment date", value=date(2025, 6, 1))
        shoe_size = st.number_input(
            "Shoe size", min_value=0.0, max_value=20.0, value=9.0,
            help="Deliberately irrelevant to the prediction -- kept to show the model ignores it.",
        )
        lucky_number = st.number_input(
            "Lucky number", min_value=1, max_value=100, value=42, step=1,
            help="Also deliberately irrelevant -- pure noise, same reason as above.",
        )

    submitted = st.form_submit_button("Predict final score", type="primary")

# ---- call the API, show the result --------------------------------------------------------
if submitted:
    payload = {
        "study_hours": study_hours,
        "attendance_pct": attendance_pct,
        "mock_test_1": mock_test_1,
        "mock_test_2": mock_test_2,
        "mock_test_3": mock_test_3,
        "income_bracket": income_bracket,
        "city": city,
        "enrollment_date": enrollment_date.isoformat(),
        "shoe_size": shoe_size,
        "lucky_number": int(lucky_number),
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}: {e}")
    else:
        if response.status_code == 200:
            result = response.json()
            score = result["predicted_final_score"]
            st.metric("Predicted final score", f"{score:.1f} / 100")
            st.caption(f"model_version: {result.get('model_version', 'n/a')}  |  "
                       f"request_id: {result.get('request_id', 'n/a')}")
            if score >= 75:
                st.success("On track.")
            elif score >= 50:
                st.warning("Worth a check-in -- borderline.")
            else:
                st.error("At risk -- this is exactly the kind of student this project was built to flag early.")
        else:
            # The API validates input with Pydantic -- a 422 here means something in the
            # form violates the schema (shouldn't normally happen through this UI, since
            # the widgets above already constrain most values, but real API clients won't
            # always be this well-behaved, so we still surface it honestly).
            st.error(f"API returned {response.status_code}: {response.text}")

st.divider()
st.caption(
    "This form sends raw, unprocessed values -- exactly what a real caller would send. "
    "All feature engineering (imputation, scaling, encoding, PCA, feature selection) "
    "happens inside the API's saved pipeline, not in this UI."
)
