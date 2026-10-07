"""
features.py

WHY does this file exist, separate from the notebook?
    When you `joblib.dump()` a pipeline that contains a CUSTOM class (like the
    FeatureCreator below), joblib doesn't save the class's actual code — it only saves a
    REFERENCE to where that class lives (its module path). If that class was defined
    inline in a notebook, its "module path" is the notebook's temporary __main__
    namespace, which doesn't exist anymore once the notebook process ends. Any OTHER
    process trying to load the pickle later (a test script, the FastAPI app, a teammate's
    machine) will fail with an error like:

        AttributeError: Can't get attribute 'FeatureCreator' on <module '__main__'>

    The fix is simple once you know it: define custom classes in a real, importable .py
    file — like this one — that both the notebook (which imports it to build the
    pipeline) and every consumer of the pickle (the FastAPI app, tests, anything else)
    can import from the SAME place. This is one of the most common real "it worked in my
    notebook but broke in production" mistakes, and it's exactly why this file exists.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCreator(BaseEstimator, TransformerMixin):
    """Creates tenure_days and study_x_attendance from raw columns.
    No fitting required -- every value here is computed independently, per row."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["tenure_days"] = (
            pd.Timestamp("2026-01-01") - pd.to_datetime(X["enrollment_date"])
        ).dt.days
        X["study_x_attendance"] = X["study_hours"] * X["attendance_pct"]
        X = X.drop(columns=["enrollment_date"])
        return X
