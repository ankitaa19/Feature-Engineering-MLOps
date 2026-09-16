"""
src/feature_engineering/dimensionality_reduction.py

WHY?
    PCA finds the directions of greatest variance in your data -- but "greatest variance"
    is meaningless if your features are on wildly different scales (a column in rupees vs.
    a column in years would make PCA think the rupee column is "more important" purely
    because its numbers are bigger). PCA must run AFTER scaling, every time.

WHAT?
    `build_pca_pipeline()` returns a scikit-learn `Pipeline` that scales first, then applies
    PCA -- so you can never accidentally run PCA on unscaled data through this framework.

HOW?
    from src.feature_engineering.dimensionality_reduction import build_pca_pipeline

    # Example chain: Scaling -> PCA -> Logistic Regression
    pca_pipeline = build_pca_pipeline(n_components=0.95)
    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),   # impute + encode raw columns
        ("pca", pca_pipeline),             # scale (again, safely) + reduce
        ("model", get_model("logistic")),
    ])
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def build_pca_pipeline(n_components: int | float = 0.95, scale_first: bool = True, random_state: int = 42) -> Pipeline:
    """Build a (scale ->) PCA pipeline.

    Parameters
    ----------
    n_components : int or float, default 0.95
        - int: keep exactly this many components.
        - float in (0, 1): keep as many components as needed to explain at least this
          fraction of total variance (0.95 = keep enough for 95% explained variance).
    scale_first : bool, default True
        If True (recommended), standardize features before PCA. Only set this False if
        your input is ALREADY scaled (e.g. it's the output of another pipeline step).
    random_state : int, default 42
        For reproducibility of PCA's internal solver.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = []
    if scale_first:
        steps.append(("scaler", StandardScaler()))
    steps.append(("pca", PCA(n_components=n_components, random_state=random_state)))
    return Pipeline(steps)
