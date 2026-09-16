"""
src/models/factory.py

WHY?
    This is the ONE function you actually import in your project code. You shouldn't have
    to remember whether "random forest" lives in the regression registry or the
    classification one, or whether it's spelled `RandomForestRegressor` or
    `RandomForestClassifier` -- `get_model()` merges both registries into one lookup and
    hands you back a ready-to-fit, correctly-parameterized model.

WHAT?
    `get_model(name, **hyperparameters)` looks up `name` in a combined registry (built from
    `REGRESSION_MODELS` and `CLASSIFICATION_MODELS`) and returns a NEW instance of that model
    class, with any hyperparameters you pass through directly to the constructor.

HOW?
    from src.models.factory import get_model

    model = get_model("ridge", alpha=1.0)
    model = get_model("random_forest_classifier", n_estimators=300, max_depth=6, random_state=42)
"""
from src.models.regression import REGRESSION_MODELS
from src.models.classification import CLASSIFICATION_MODELS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# A single combined registry -- a plain dict, deliberately not a big if/else chain,
# so adding a new model is a one-line addition, not a new branch to maintain.
_MODEL_REGISTRY = {**REGRESSION_MODELS, **CLASSIFICATION_MODELS}


def list_available_models() -> dict[str, list[str]]:
    """Return the available model names, split by task, for discoverability."""
    return {
        "regression": sorted(REGRESSION_MODELS.keys()),
        "classification": sorted(CLASSIFICATION_MODELS.keys()),
    }


def get_model(name: str, **hyperparameters):
    """Instantiate a model by name.

    Parameters
    ----------
    name : str
        Registered model name. Call `list_available_models()` to see all options.
    **hyperparameters :
        Passed directly to the model's constructor, e.g. `get_model("ridge", alpha=0.5)`.

    Returns
    -------
    A new, unfit scikit-learn estimator instance.

    Raises
    ------
    ValueError
        If `name` is not a registered model. The error message lists the valid names,
        so you don't have to go looking for them.
    """
    if name not in _MODEL_REGISTRY:
        available = list_available_models()
        raise ValueError(
            f"Unknown model '{name}'.\n"
            f"Available regression models: {available['regression']}\n"
            f"Available classification models: {available['classification']}"
        )

    model_class = _MODEL_REGISTRY[name]
    model = model_class(**hyperparameters)
    logger.info(f"Created model: {name} -> {model_class.__name__}({hyperparameters})")
    return model


def register_model(name: str, model_class, task: str = "regression") -> None:
    """Register a NEW model under a custom name, so `get_model()` can find it too.

    Parameters
    ----------
    name : str
        The name you'll use to call `get_model(name)`.
    model_class : class
        Any scikit-learn-compatible estimator class (must accept **kwargs in __init__
        and implement .fit()/.predict()).
    task : {"regression", "classification"}, default "regression"
        Which registry to add it to -- purely for organization; `get_model()` looks in
        the combined registry regardless.

    Example
    -------
    from xgboost import XGBRegressor
    register_model("xgboost", XGBRegressor, task="regression")
    model = get_model("xgboost", n_estimators=200)
    """
    if task == "regression":
        REGRESSION_MODELS[name] = model_class
    elif task == "classification":
        CLASSIFICATION_MODELS[name] = model_class
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got '{task}'.")
    _MODEL_REGISTRY[name] = model_class
    logger.info(f"Registered new model '{name}' -> {model_class.__name__} ({task})")
