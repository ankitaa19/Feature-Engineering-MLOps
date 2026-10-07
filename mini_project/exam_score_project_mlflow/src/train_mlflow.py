"""src/train_mlflow.py

The exact same pipeline as notebooks/exam_score_pipeline.ipynb (same columns, same
transformers, same model). Nothing about the FEATURE ENGINEERING changes here -- the only
thing added is MLflow *tracking*: every time this script runs, it records the parameters
used, the metrics produced, and the trained model itself as one "run" inside an MLflow
experiment, instead of those things only existing as print() output that scrolls away.

Run it multiple times with different --lasso-alpha values to build up a small set of
comparable experiments:

    python src/train_mlflow.py --lasso-alpha 0.3 --run-name "alpha_0.3_baseline"
    python src/train_mlflow.py --lasso-alpha 0.6 --run-name "alpha_0.6"
    python src/train_mlflow.py --lasso-alpha 0.1 --run-name "alpha_0.1"

Then compare them either from the command line (see MLFLOW_GUIDE.md) or visually in the
MLflow UI:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features import FeatureCreator

EXPERIMENT_NAME = "exam_score_prediction"
REGISTERED_MODEL_NAME = "exam_score_predictor"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the exam-score pipeline with MLflow tracking.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=1)
    parser.add_argument("--lasso-alpha", type=float, default=0.3)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--register", action="store_true",
                         help="Also register the trained model in the MLflow Model Registry.")
    return parser.parse_args()


def build_pipeline(pca_components: int, lasso_alpha: int, random_state: int) -> Pipeline:
    numeric_cols = ["study_hours", "attendance_pct", "shoe_size", "lucky_number", "tenure_days", "study_x_attendance"]
    mock_test_cols = ["mock_test_1", "mock_test_2", "mock_test_3"]
    nominal_cols = ["city"]
    ordinal_cols = ["income_bracket"]

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    mock_test_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=pca_components, random_state=random_state)),
    ])
    nominal_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[["Low", "Medium", "High"]])),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("mock_tests_pca", mock_test_pipe, mock_test_cols),
        ("nominal", nominal_pipe, nominal_cols),
        ("ordinal", ordinal_pipe, ordinal_cols),
    ])

    return Pipeline([
        ("feature_creation", FeatureCreator()),
        ("preprocessing", preprocessor),
        ("feature_selection", SelectFromModel(Lasso(alpha=lasso_alpha, random_state=random_state))),
        ("model", LinearRegression()),
    ])


def main():
    args = parse_args()

    # 1. Point MLflow at a SQLite-backed tracking store (not the default plain-file store).
    #    Why SQLite and not the default "mlruns/" folder store? The Model Registry (Step 4)
    #    needs a real database backend -- the plain file store cannot register models.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv("data/raw/student_exam_scores.csv", parse_dates=["enrollment_date"])
    X = df.drop(columns=["student_id", "final_score"])
    y = df["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    with mlflow.start_run(run_name=args.run_name):
        # 2. Log every input that could change the result -- exactly the numbers that used
        #    to be hard-coded, now visible as columns in the MLflow UI/comparison table.
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("pca_components", args.pca_components)
        mlflow.log_param("lasso_alpha", args.lasso_alpha)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))

        pipeline = build_pipeline(args.pca_components, args.lasso_alpha, args.random_state)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Test R^2: {r2:.3f}")
        print(f"Test MAE: {mae:.2f}")

        # 3. Log the outcome. This is what makes runs COMPARABLE -- every run in this
        #    experiment now has an r2 and mae column, sortable in the UI.
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # 4. Log the model ITSELF as an MLflow artifact (not just the metrics about it).
        #    serialization_format="pickle" avoids MLflow's newer default "skops" format,
        #    which refuses to load custom classes like our FeatureCreator without an
        #    explicit trust flag -- plain pickle matches how the rest of this project
        #    already saves/loads the pipeline (joblib, which is pickle-based).
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="pickle",
            registered_model_name=REGISTERED_MODEL_NAME if args.register else None,
        )

        # Also keep saving the plain joblib file, so the existing FastAPI/Streamlit
        # deployment code (which loads models/exam_score_pipeline.joblib directly) is
        # completely undisturbed by adding MLflow on top.
        joblib.dump(pipeline, "models/exam_score_pipeline.joblib")

        run = mlflow.active_run()
        print(f"Logged to MLflow run: {run.info.run_id}")
        print(f"Experiment: {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
