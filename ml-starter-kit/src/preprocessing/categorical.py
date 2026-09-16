"""
src/preprocessing/categorical.py

WHY?
    Models need numbers, not text -- "Mumbai" and "Delhi" mean nothing to a linear model or
    a gradient boosted tree until they're converted into numeric columns. And whatever encoder
    you fit on training data WILL eventually see a category it's never seen before at
    inference time (a new city, a new product code) -- if that isn't handled safely, your
    pipeline crashes in production instead of just handling it gracefully.

WHAT?
    `build_categorical_pipeline()` returns a scikit-learn `Pipeline` that imputes missing
    categorical values and then encodes the result, using one-hot or ordinal encoding.

HOW?
    from src.preprocessing.categorical import build_categorical_pipeline
    categorical_pipe = build_categorical_pipeline(encoding="onehot")
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

_IMPUTATION_STRATEGIES = {None, "most_frequent", "constant"}
_ENCODERS = {"onehot", "ordinal"}


def build_categorical_pipeline(
    imputation: str | None = "most_frequent",
    encoding: str = "onehot",
    fill_value: str = "missing",
) -> Pipeline:
    """Build a categorical preprocessing pipeline: impute, then encode.

    Parameters
    ----------
    imputation : {"most_frequent", "constant", None}, default "most_frequent"
        Strategy for filling missing categorical values. "constant" fills with `fill_value`.
    encoding : {"onehot", "ordinal"}, default "onehot"
        - "onehot": one binary column per category (no assumed order). Uses
          `handle_unknown="ignore"` so a category never seen during training is encoded as
          all-zeros at inference time, instead of raising an error.
        - "ordinal": one integer column per feature (assumes a meaningful order -- only use
          this for genuinely ordinal data like "low/medium/high", not nominal categories
          like city names).
    fill_value : str, default "missing"
        The constant used to fill missing values when `imputation="constant"`.

    Returns
    -------
    sklearn.pipeline.Pipeline

    Raises
    ------
    ValueError
        If `imputation` or `encoding` is not one of the supported options.
    """
    if imputation not in _IMPUTATION_STRATEGIES:
        raise ValueError(f"imputation must be one of {_IMPUTATION_STRATEGIES}, got '{imputation}'.")
    if encoding not in _ENCODERS:
        raise ValueError(f"encoding must be one of {_ENCODERS}, got '{encoding}'.")

    steps = []
    if imputation == "constant":
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=fill_value)))
    elif imputation is not None:
        steps.append(("imputer", SimpleImputer(strategy=imputation)))

    if encoding == "onehot":
        steps.append(("encoder", OneHotEncoder(handle_unknown="ignore")))
    else:  # ordinal
        steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

    return Pipeline(steps)
