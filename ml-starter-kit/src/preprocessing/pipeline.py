"""
src/preprocessing/pipeline.py

WHY?
    Numeric and categorical columns need different treatment, but a model needs them
    combined into one feature matrix. Doing this by hand every time (slice numeric columns,
    slice categorical columns, transform each, concatenate) is exactly the kind of boilerplate
    this framework exists to remove -- and doing it wrong (e.g. fitting a scaler on the
    combined data before splitting) is how preprocessing leakage happens.

WHAT?
    `build_preprocessor()` returns a scikit-learn `ColumnTransformer` that routes numeric
    columns through `build_numeric_pipeline()` and categorical columns through
    `build_categorical_pipeline()`, then concatenates the results into one feature matrix.

    `build_pipeline()` chains that preprocessor with a model into ONE final
    `sklearn.Pipeline`, so `.fit()` and `.predict()` handle raw data end to end -- this is
    what actually prevents preprocessing leakage structurally: the pipeline can only ever be
    fit on whatever data you explicitly hand it.

HOW?
    from src.preprocessing.pipeline import build_preprocessor, build_pipeline

    preprocessor = build_preprocessor(
        numerical_features=["age", "income"],
        categorical_features=["city", "contract_type"],
    )
    pipeline = build_pipeline(preprocessor, model)
    pipeline.fit(X_train, y_train)   # preprocessor is fit ONLY on X_train, guaranteed
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.preprocessing.numerical import build_numeric_pipeline
from src.preprocessing.categorical import build_categorical_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    numeric_strategy: str | None = "median",
    scaler: str | None = "standard",
    categorical_encoder: str = "onehot",
    categorical_imputation: str | None = "most_frequent",
) -> ColumnTransformer:
    """Build a ColumnTransformer that preprocesses numeric and categorical columns separately.

    Parameters
    ----------
    numerical_features : list[str]
        Column names to treat as numeric.
    categorical_features : list[str]
        Column names to treat as categorical.
    numeric_strategy : str, default "median"
        Imputation strategy for numeric columns. See `build_numeric_pipeline`.
    scaler : str, default "standard"
        Scaling strategy for numeric columns. See `build_numeric_pipeline`.
    categorical_encoder : str, default "onehot"
        Encoding strategy for categorical columns. See `build_categorical_pipeline`.
    categorical_imputation : str, default "most_frequent"
        Imputation strategy for categorical columns. See `build_categorical_pipeline`.

    Returns
    -------
    sklearn.compose.ColumnTransformer

    Raises
    ------
    ValueError
        If `numerical_features` and `categorical_features` overlap, or if both are empty.
    """
    overlap = set(numerical_features) & set(categorical_features)
    if overlap:
        raise ValueError(
            f"Columns cannot be both numerical and categorical: {sorted(overlap)}"
        )
    if not numerical_features and not categorical_features:
        raise ValueError("At least one of numerical_features or categorical_features must be non-empty.")

    transformers = []
    if numerical_features:
        numeric_pipe = build_numeric_pipeline(imputation=numeric_strategy, scaling=scaler)
        transformers.append(("numeric", numeric_pipe, numerical_features))
    if categorical_features:
        categorical_pipe = build_categorical_pipeline(
            imputation=categorical_imputation, encoding=categorical_encoder
        )
        transformers.append(("categorical", categorical_pipe, categorical_features))

    logger.info(
        f"Built preprocessor: {len(numerical_features)} numeric, "
        f"{len(categorical_features)} categorical columns"
    )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Chain a preprocessor and a model into a single, fittable/predictable Pipeline.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Typically the output of `build_preprocessor()`.
    model : sklearn estimator
        Typically the output of `get_model()` from `src.models.factory`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with two steps: "preprocessor" then "model". Calling `.fit(X_train, y_train)`
        fits BOTH steps together, using only the data you pass in -- this is what makes
        preprocessing leakage structurally impossible as long as you always fit on the
        training split only.
    """
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
