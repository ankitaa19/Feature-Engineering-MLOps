"""
src/utils/io.py

WHY?
    A trained pipeline is worthless if you can't get it back later -- and re-training from
    scratch every time you want to make a prediction isn't practical. Saving the WHOLE
    pipeline (preprocessing + model together, as one object) also guarantees train/inference
    consistency automatically: whatever preprocessing the model learned during `.fit()` is
    baked into the same object that makes predictions later, so there's no separate
    "prediction-time preprocessing code" to accidentally get out of sync.

WHAT?
    - `save_model()` / `load_model()`: persist and restore a fitted pipeline with joblib.
    - `predict()` / `predict_proba()`: thin, explicit wrappers that make it obvious you're
      expected to pass RAW input data -- the pipeline handles imputation, scaling, encoding,
      and any feature transformation internally, exactly as it did during training.

HOW?
    from src.utils.io import save_model, load_model, predict

    save_model(pipeline, "models/churn_model.joblib")
    loaded = load_model("models/churn_model.joblib")
    predictions = predict(loaded, new_raw_dataframe)
"""
from pathlib import Path
import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_model(pipeline, path: str) -> None:
    """Save a fitted pipeline (or any picklable object) to disk with joblib.

    Parameters
    ----------
    pipeline : fitted sklearn estimator/Pipeline
        Save the WHOLE pipeline, not just the model step -- that's what guarantees
        identical preprocessing at inference time.
    path : str
        Destination path. Parent directories are created automatically if missing.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, file_path)
    logger.info(f"Saved model to '{file_path}'")


def load_model(path: str):
    """Load a previously saved pipeline.

    Parameters
    ----------
    path : str

    Returns
    -------
    The fitted pipeline object, ready to call `.predict()` on raw input data.

    Raises
    ------
    FileNotFoundError
        If no file exists at `path`.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No saved model found at '{path}'.")
    pipeline = joblib.load(file_path)
    logger.info(f"Loaded model from '{file_path}'")
    return pipeline


def predict(pipeline, X_new):
    """Predict on RAW, unprocessed input data.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
        The pipeline's preprocessing step handles imputation/scaling/encoding internally --
        do NOT preprocess `X_new` yourself before calling this.
    X_new : pd.DataFrame or array-like
        Raw feature data, in the same column structure as training data (same column
        names as whatever `build_preprocessor()` was told to expect).

    Returns
    -------
    np.ndarray of predictions.
    """
    return pipeline.predict(X_new)


def predict_proba(pipeline, X_new):
    """Predict class probabilities on RAW, unprocessed input data (classification only).

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
        Must have a classifier as its final step (i.e. expose `.predict_proba()`).
    X_new : pd.DataFrame or array-like
        Raw feature data.

    Returns
    -------
    np.ndarray of shape (n_samples, n_classes).

    Raises
    ------
    AttributeError
        If the pipeline's model doesn't support `.predict_proba()` (e.g. it's a regressor,
        or a classifier that doesn't implement probability estimates).
    """
    if not hasattr(pipeline, "predict_proba"):
        raise AttributeError(
            "This pipeline's model does not support predict_proba() -- "
            "it's either a regressor, or a classifier without probability support."
        )
    return pipeline.predict_proba(X_new)
