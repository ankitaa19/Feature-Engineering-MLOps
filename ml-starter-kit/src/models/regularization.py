"""
src/models/regularization.py

WHY?
    Ridge, Lasso, and ElasticNet are already available through `get_model("ridge")` etc.
    (see `src.models.regression`) -- but regularized linear models come with one extra
    question that plain models don't: "how does the coefficient change as I vary the
    penalty strength?" That's a genuinely different, recurring need (choosing alpha,
    explaining Lasso's feature-selection behavior to a stakeholder, building a
    regularization-path plot) that deserves its own small toolkit rather than being
    bolted onto the model registry.

WHAT?
    - `DEFAULT_ALPHAS`: a sensible log-spaced grid of penalty strengths to sweep for
      GridSearchCV / plotting, so you don't have to invent one from scratch every time.
    - `compute_regularization_path()`: returns each coefficient's value across a range of
      alphas for Ridge or Lasso, ready to plot.

HOW?
    from src.models.regularization import compute_regularization_path, DEFAULT_ALPHAS
    alphas, coefs = compute_regularization_path(X_train, y_train, method="lasso")
"""
import numpy as np
from sklearn.linear_model import Ridge, lasso_path

DEFAULT_ALPHAS = np.logspace(-3, 2, 30)  # 0.001 to 100, log-spaced -- a reasonable default sweep


def compute_regularization_path(X, y, method: str = "lasso", alphas=None, max_iter: int = 5000):
    """Compute how each feature's coefficient changes across a range of penalty strengths.

    Parameters
    ----------
    X : array-like
        Standardized feature matrix (regularization paths are only meaningful when
        features are on comparable scales -- always scale before calling this).
    y : array-like
        Target values. Should ALSO be standardized/scaled to a comparable variance --
        alpha's effective strength is relative to the target's scale, not just the
        features'. Skipping this makes it hard to pick an alpha range that shows any
        meaningful variation at all.
    method : {"lasso", "ridge"}, default "lasso"
        Which regularized model's path to trace.
    alphas : array-like, optional
        Penalty strengths to evaluate. Defaults to `DEFAULT_ALPHAS`.
    max_iter : int, default 5000
        Maximum coordinate-descent iterations for the Lasso path (ignored for Ridge, which
        has a closed-form solution). Datasets with strongly collinear/near-duplicate
        features can need more than scikit-learn's default 1000 iterations to fully
        converge, especially at small alpha values -- raise this further if you still see
        `ConvergenceWarning`.

    Returns
    -------
    alphas : np.ndarray, shape (n_alphas,)
        NOTE: `lasso_path` always returns alphas sorted from LARGEST to SMALLEST
        regardless of the order you pass in -- `coefs[:, 0]` corresponds to the most
        heavily regularized end, `coefs[:, -1]` to the least. Always index by matching
        `alphas[i]` to `coefs[:, i]` rather than assuming input order is preserved.
    coefs : np.ndarray, shape (n_features, n_alphas)
        `coefs[j, i]` is feature j's coefficient at `alphas[i]`.

    Raises
    ------
    ValueError
        If `method` is not "lasso" or "ridge".
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    if method == "lasso":
        computed_alphas, coefs, _ = lasso_path(X, y, alphas=alphas, max_iter=max_iter)
        return computed_alphas, coefs
    elif method == "ridge":
        coefs = []
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X, y)
            coefs.append(model.coef_)
        return np.array(alphas), np.array(coefs).T
    else:
        raise ValueError(f"method must be 'lasso' or 'ridge', got '{method}'.")
