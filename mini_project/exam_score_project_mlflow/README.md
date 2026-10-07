# Student Exam Score Predictor — Notebook → Pickle → FastAPI

One dataset. Every major feature engineering concept covered once. One deliberately simple
model (Linear Regression) so the focus stays on the FEATURES, not on an opaque algorithm.
Then: the exact path from a notebook to a real, tested, running application.

## What this project demonstrates

| Concept | Where |
|---|---|
| Train/test split *before* any preprocessing | `notebooks/exam_score_pipeline.ipynb`, Section 2 |
| Missing value imputation | Section 4 (`study_hours` has real missing values) |
| Feature creation (time-based + interaction) | Section 3 (`tenure_days`, `study_x_attendance`) |
| Categorical encoding (nominal + ordinal) | Section 4 (`city` one-hot, `income_bracket` ordinal) |
| Scaling | Section 4 |
| Dimensionality reduction (PCA) | Section 5 — on 3 genuinely correlated mock test scores |
| Feature selection (embedded, Lasso) | Section 6 — automatically drops 2 planted noise columns |
| One leakage-safe `sklearn.Pipeline` | Section 8 |
| Saving the pipeline (pickling) | Section 11 — `joblib`, with a real bug caught and fixed (see below) |
| Notebook → application | `app/main.py` — a FastAPI service |
| Testing | `tests/test_pipeline.py`, `tests/test_api.py` |

## A real bug this project caught (and how it was fixed)

The pipeline includes a custom `FeatureCreator` transformer. Defining it directly inside the
notebook works fine for training — but `joblib.dump()` only saves a *reference* to where a
custom class lives, not its code. Loading the pickle from anywhere else (a test, the FastAPI
app) then fails with:

```
AttributeError: Can't get attribute 'FeatureCreator' on <module '__main__'>
```

**The fix:** `FeatureCreator` now lives in its own importable file, `features.py`, at the
project root. Both the notebook and `app/main.py` import it from there. This is one of the
most common real "worked in my notebook, broke in production" mistakes — see `features.py`'s
docstring and the notebook's own callout for the full explanation.

## Project structure

```
exam_score_project/
├── data/raw/
│   ├── generate_data.py            # regenerate the dataset (fixed seed)
│   └── student_exam_scores.csv
├── notebooks/
│   └── exam_score_pipeline.ipynb   # the full, executed, end-to-end notebook
├── features.py                     # FeatureCreator -- imported by BOTH the notebook and the app
├── app/
│   └── main.py                     # FastAPI application
├── models/
│   └── exam_score_pipeline.joblib  # the trained, saved pipeline
├── tests/
│   ├── test_pipeline.py            # tests the ML pipeline directly
│   └── test_api.py                 # tests the FastAPI layer
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the notebook (trains and saves the model)

```bash
python data/raw/generate_data.py          # regenerate the dataset, if needed
jupyter notebook notebooks/exam_score_pipeline.ipynb
```

Running it end to end produces `models/exam_score_pipeline.joblib`.

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Then, in another terminal:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
    "study_hours": 12.5, "attendance_pct": 88.0,
    "mock_test_1": 72.0, "mock_test_2": 75.0, "mock_test_3": 70.0,
    "income_bracket": "Medium", "city": "Pune",
    "enrollment_date": "2025-06-01", "shoe_size": 9.0, "lucky_number": 42
}'
# -> {"predicted_final_score": 95.2}
```

Or open `http://localhost:8000/docs` for FastAPI's interactive Swagger UI — it's generated
automatically from the same Pydantic model that validates real requests.

## Run the tests

```bash
pytest tests/ -v
```

16 tests, split deliberately into two files:
- **`test_pipeline.py`** (7 tests) — the ML pipeline directly: loads correctly, predicts in a
  sane range, handles missing values and unseen categories without crashing, more study
  hours predicts a higher score (directional sanity), deterministic output, and the two
  planted noise columns (`shoe_size`, `lucky_number`) genuinely don't move the prediction.
- **`test_api.py`** (9 tests) — the FastAPI layer: health check, valid request → 200,
  response shape, API's prediction matches calling the pipeline directly (the endpoint adds
  zero logic of its own), and proper 422 validation errors for missing fields, invalid
  categories, and out-of-range values.

## Why Linear Regression?

Every surviving feature after selection has ONE coefficient with a direct, readable meaning
— "holding everything else fixed, how many points does one unit of this feature add?" That
keeps the notebook's focus where it belongs for a feature-engineering project: on whether the
FEATURES are good, not on decoding an opaque model.

## Beyond `uvicorn --reload`: Docker, a Streamlit UI, and AWS

Everything above runs on your own laptop, with you keeping a terminal open. The next module —
**`DOCKER_STREAMLIT_AWS_GUIDE.md`** — takes this exact same pipeline and API all the way to a
real URL that stays up without your laptop:

| Stage | What it adds | File(s) |
|---|---|---|
| 1. Docker basics | Package the API alone into a portable image | `docker/Dockerfile.api-only` |
| 2. Streamlit UI | A form-based UI that calls the API over HTTP (no second copy of the model) | `streamlit_app.py` |
| 3. Bundle both | ONE image running both services via a small supervisor script | `Dockerfile`, `docker/entrypoint.sh` |
| 4. Docker Hub | Push the image so any machine (including AWS) can pull it | — |
| 5. AWS EC2 | A real, always-on server running the container | `deploy/aws/ec2-user-data.sh`, `deploy/aws/EC2_WALKTHROUGH.md` |

No new ML logic is introduced anywhere in this module — same pipeline, same API, same
predictions. Only how it's *packaged and run* changes. Start with
`DOCKER_STREAMLIT_AWS_GUIDE.md`.
