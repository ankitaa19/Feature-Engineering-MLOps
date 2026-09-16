"""
src/preprocessing/numerical.py

WHY?
    Numeric columns almost always need two things before a model can use them well:
    missing values filled in, and values put on a comparable scale (so a model doesn't
    treat "income in rupees" as automatically more important than "age in years" just
    because its raw numbers are bigger). Every project needs this; almost no project
    needs a different way of doing it.

WHAT?
    `build_numeric_pipeline()` returns a scikit-learn `Pipeline` that imputes missing
    values and then scales the result, using whichever strategy you choose.

HOW?
    from src.preprocessing.numerical import build_numeric_pipeline
    numeric_pipe = build_numeric_pipeline(imputation="median", scaling="standard")
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

_IMPUTATION_STRATEGIES = {None, "mean", "median", "most_frequent"}
_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    None: None,
}


def build_numeric_pipeline(imputation: str | None = "median", scaling: str | None = "standard") -> Pipeline:
    """Build a numeric preprocessing pipeline: impute, then scale.

    Parameters
    ----------
    imputation : {"mean", "median", "most_frequent", None}, default "median"
        Strategy for filling missing numeric values. `None` skips imputation entirely
        (only safe if you're certain there are no missing values).
    scaling : {"standard", "minmax", "robust", None}, default "standard"
        - "standard": zero mean, unit variance (StandardScaler) -- the usual default.
        - "minmax": rescale into [0, 1] (MinMaxScaler) -- useful when you need bounded values.
        - "robust": scale using median/IQR (RobustScaler) -- more resistant to outliers.
        - None: skip scaling (rarely a good idea for distance-based or gradient-based models).

    Returns
    -------
    sklearn.pipeline.Pipeline

    Raises
    ------
    ValueError
        If `imputation` or `scaling` is not one of the supported options.
    """
    if imputation not in _IMPUTATION_STRATEGIES:
        raise ValueError(f"imputation must be one of {_IMPUTATION_STRATEGIES}, got '{imputation}'.")
    if scaling not in _SCALERS:
        raise ValueError(f"scaling must be one of {sorted(k for k in _SCALERS if k)} or None, got '{scaling}'.")

    steps = []
    if imputation is not None:
        steps.append(("imputer", SimpleImputer(strategy=imputation)))
    if scaling is not None:
        steps.append(("scaler", _SCALERS[scaling]()))

    if not steps:
        # sklearn Pipelines need at least one step; a passthrough identity step keeps the
        # interface consistent even if the caller opted out of both imputation and scaling.
        from sklearn.preprocessing import FunctionTransformer
        steps.append(("passthrough", FunctionTransformer()))

    return Pipeline(steps)
