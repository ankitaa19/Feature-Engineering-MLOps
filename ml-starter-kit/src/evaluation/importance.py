"""
src/evaluation/importance.py

WHY?
    "Which features matter?" is one of the most common questions asked of any model --
    but answering it correctly requires two things people often get wrong: (1) using more
    than one method, since built-in importance, permutation importance, and SHAP can
    disagree; and (2) correctly tracking WHAT each coefficient/importance value actually
    refers to after a ColumnTransformer, a OneHotEncoder that split one column into many,
    or a PCA step that replaced your original columns with abstract components entirely.

WHAT?
    - `get_feature_names(pipeline)`: correctly reconstructs feature names after
      preprocessing -- expanding one-hot-encoded columns into their real category names,
      and explicitly labeling PCA output as "PC1, PC2, ..." rather than pretending those
      are still your original columns (they are NOT -- each principal component is a
      blend of many original features, and has no single original name).
    - `tree_feature_importance()`: built-in importances from a fitted tree-based model.
    - `permutation_feature_importance()`: shuffle-and-measure importances, model-agnostic.
    - `shap_feature_importance()`: SHAP values, if the `shap` package is installed
      (optional dependency -- raises a clear, actionable error if it isn't).

HOW?
    from src.evaluation.importance import get_feature_names, tree_feature_importance

    names = get_feature_names(fitted_pipeline)
    importances = tree_feature_importance(fitted_pipeline, feature_names=names)
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_feature_names(pipeline) -> list[str]:
    """Correctly recover feature names after a fitted pipeline's preprocessing steps.

    Handles three cases explicitly:
    1. A plain ColumnTransformer with imputers/scalers/encoders -- uses each transformer's
       own `get_feature_names_out()`, so a one-hot-encoded "city" column correctly becomes
       "city_Mumbai", "city_Delhi", etc., not just "city" repeated.
    2. A PCA step anywhere in the pipeline -- returns "PC1", "PC2", ... instead. This is
       deliberate: a principal component is a weighted blend of MANY original features, so
       it has no single "real" name, and pretending otherwise (e.g. reusing the original
       column names) would misrepresent what the model is actually using.
    3. No preprocessing step found -- falls back to generic "feature_0", "feature_1", ...

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
        Expected to contain a step named "preprocessor" (from `build_preprocessor`) and,
        optionally, a step named "pca" (from `build_pca_pipeline`).

    Returns
    -------
    list[str]
    """
    if hasattr(pipeline, "named_steps") and "pca" in getattr(pipeline, "named_steps", {}):
        n_components = pipeline.named_steps["pca"].named_steps["pca"].n_components_
        logger.info(f"Pipeline includes PCA -- returning {n_components} generic component names, not original feature names.")
        return [f"PC{i+1}" for i in range(n_components)]

    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            names = list(preprocessor.get_feature_names_out())
            # ColumnTransformer prefixes names with the transformer name (e.g. "numeric__age") --
            # strip that prefix for readability, since it's rarely useful downstream.
            cleaned = [n.split("__", 1)[-1] for n in names]
            return cleaned

    logger.warning("Could not determine feature names from pipeline structure -- using generic names.")
    return None  # let the caller fall back to generic names sized to their actual data


def tree_feature_importance(fitted_model, feature_names: list[str] = None) -> pd.DataFrame:
    """Extract built-in importances from a fitted tree-based model (or the "model" step
    of a fitted Pipeline).

    Parameters
    ----------
    fitted_model : fitted sklearn estimator or Pipeline
        Must expose `.feature_importances_` (tree-based models) after fitting -- or be a
        Pipeline whose "model" step does.
    feature_names : list[str], optional
        Names to attach to each importance value. If not given, generic names are used.
        Get correct names with `get_feature_names(pipeline)` first if your pipeline has
        preprocessing steps.

    Returns
    -------
    pd.DataFrame with columns ["feature", "importance"], sorted descending.

    Raises
    ------
    AttributeError
        If the model doesn't expose `.feature_importances_` (e.g. it's a linear model --
        use `.coef_` directly, or use `permutation_feature_importance` instead, which
        works for any model).
    """
    model = fitted_model.named_steps["model"] if hasattr(fitted_model, "named_steps") else fitted_model

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} has no `.feature_importances_` attribute. "
            f"This method only works for tree-based models. For linear models, use "
            f"`.coef_` directly; for a model-agnostic option, use `permutation_feature_importance`."
        )

    importances = model.feature_importances_
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_feature_importance(
    fitted_pipeline, X, y, feature_names: list[str] = None, n_repeats: int = 10,
    scoring: str = None, random_state: int = 42,
) -> pd.DataFrame:
    """Compute model-agnostic permutation importance: shuffle one feature at a time and
    measure how much the score drops.

    IMPORTANT -- match your feature names to your data's granularity:
    If you pass a FULL pipeline (preprocessing + model) together with RAW X, importance is
    computed per RAW column, so use `X.columns` (or similarly raw-level names) -- NOT the
    expanded post-one-hot names from `get_feature_names()`. If instead you want importance
    at the EXPANDED, post-preprocessing feature level (e.g. one importance per one-hot
    category), pass just the fitted MODEL step together with ALREADY-PREPROCESSED X:
        X_processed = fitted_pipeline.named_steps["preprocessor"].transform(X)
        permutation_feature_importance(fitted_pipeline.named_steps["model"], X_processed, y,
                                        feature_names=get_feature_names(fitted_pipeline))
    Mismatching these two (expanded names with raw-level data, or vice versa) raises a
    length-mismatch error -- which is deliberate: silently mislabeling importances would be
    worse than an explicit crash.

    Parameters
    ----------
    fitted_pipeline : fitted sklearn estimator or Pipeline
        Works for ANY model type, unlike `tree_feature_importance`.
    X, y : array-like
        Data to evaluate on -- typically your TEST set, so importance reflects
        genuine generalization, not memorization of the training data. Must have as many
        columns as `feature_names` (see note above).
    feature_names : list[str], optional
        Names to attach to each importance value. Must match X's column count exactly.
    n_repeats : int, default 10
        Number of times each feature is shuffled -- higher gives a more stable estimate.
    scoring : str, optional
        Scikit-learn scoring string. Defaults to the estimator's own `.score()`.
    random_state : int, default 42
        For reproducible shuffling.

    Returns
    -------
    pd.DataFrame with columns ["feature", "importance_mean", "importance_std"], sorted
    descending by importance_mean.

    Raises
    ------
    ValueError
        If `feature_names` is given but its length doesn't match the number of columns
        in `X` -- a signal that you've mismatched raw vs. preprocessed granularity (see
        the note above).
    """
    result = permutation_importance(
        fitted_pipeline, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring
    )

    if feature_names is not None and len(feature_names) != len(result.importances_mean):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X produced "
            f"{len(result.importances_mean)} importance values. This usually means you've "
            f"mixed EXPANDED (post-one-hot) feature names with RAW input data, or vice versa "
            f"-- see this function's docstring for the correct pairing of each."
        )

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(result.importances_mean))]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def shap_feature_importance(fitted_pipeline, X, feature_names: list[str] = None, max_samples: int = 200):
    """Compute SHAP values for a fitted model (optional dependency).

    Parameters
    ----------
    fitted_pipeline : fitted sklearn estimator or Pipeline
        The "model" step should be tree-based for `shap.TreeExplainer` to apply cleanly;
        other model types will fall back to `shap.Explainer`'s general-purpose handling.
    X : array-like
        ALREADY PREPROCESSED feature matrix (i.e. the output of the pipeline's
        preprocessing steps, not raw data) -- SHAP needs to see the exact numeric matrix
        the model itself consumes.
    feature_names : list[str], optional
        Names for the SHAP summary. Get correct names with `get_feature_names(pipeline)`.
    max_samples : int, default 200
        SHAP can be slow on large datasets -- only the first `max_samples` rows are used.

    Returns
    -------
    shap.Explanation object (can be passed directly to `shap.summary_plot`, etc.)

    Raises
    ------
    ImportError
        If the optional `shap` package isn't installed. The error message tells you
        exactly how to fix it.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "shap_feature_importance() requires the optional 'shap' package. "
            "Install it with: pip install shap"
        ) from e

    model = fitted_pipeline.named_steps["model"] if hasattr(fitted_pipeline, "named_steps") else fitted_pipeline
    X_sample = X[:max_samples] if hasattr(X, "__len__") else X

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model, X_sample)

    explanation = explainer(X_sample)
    if feature_names is not None:
        explanation.feature_names = feature_names
    return explanation
