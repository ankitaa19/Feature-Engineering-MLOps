"""
src/feature_engineering/transformation.py

WHY?
    Skewed numeric features (income, transaction amounts, most real-world money/count data)
    make many models -- especially linear ones -- perform worse than they should. And some
    real relationships genuinely aren't linear or additive; a feature times another feature,
    or a feature squared, can capture patterns a plain linear model can't reach on its own.

WHAT?
    - `build_power_transformer()`: wraps scikit-learn's `PowerTransformer` (Box-Cox /
      Yeo-Johnson) for fixing skew.
    - `add_polynomial_features()`: wraps `PolynomialFeatures` for interaction and
      polynomial terms.

HOW?
    from src.feature_engineering.transformation import build_power_transformer, add_polynomial_features
    transformer = build_power_transformer(method="yeo-johnson")
    poly = add_polynomial_features(degree=2, interaction_only=True)
"""
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures


def build_power_transformer(method: str = "yeo-johnson", standardize: bool = True) -> PowerTransformer:
    """Build a power transformer to reduce skew in numeric features.

    Parameters
    ----------
    method : {"yeo-johnson", "box-cox"}, default "yeo-johnson"
        "box-cox" requires strictly positive input values. "yeo-johnson" works on any
        real values (positive, negative, or zero) -- prefer it unless you specifically
        need Box-Cox's behavior and know your data is strictly positive.
    standardize : bool, default True
        If True, the transformed output is also scaled to zero mean / unit variance.

    Returns
    -------
    sklearn.preprocessing.PowerTransformer
    """
    if method not in {"yeo-johnson", "box-cox"}:
        raise ValueError(f"method must be 'yeo-johnson' or 'box-cox', got '{method}'.")
    return PowerTransformer(method=method, standardize=standardize)


def add_polynomial_features(degree: int = 2, interaction_only: bool = False, include_bias: bool = False) -> PolynomialFeatures:
    """Build a transformer that adds polynomial and/or interaction terms.

    Parameters
    ----------
    degree : int, default 2
        Highest polynomial degree to generate (e.g. degree=2 adds x1^2, x2^2, x1*x2, ...).
    interaction_only : bool, default False
        If True, only generate interaction terms (x1*x2), not pure powers (x1^2).
    include_bias : bool, default False
        If True, include a constant (bias) column of all 1s. Usually left False since
        most models already fit their own intercept.

    Returns
    -------
    sklearn.preprocessing.PolynomialFeatures

    Note
    ----
    Feature count grows fast with degree and number of input columns -- watch out for
    the curse of dimensionality (see `src.feature_engineering.selection`) if you apply
    this to more than a handful of columns.
    """
    return PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
