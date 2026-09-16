"""
src/models/regression.py

WHY?
    Trying a new regression model shouldn't mean opening a new import statement and
    remembering the exact class name every time. A registry lets you swap models by
    changing one string.

WHAT?
    A dictionary (`REGRESSION_MODELS`) mapping short, memorable names to scikit-learn
    regressor CLASSES (not instances -- they get instantiated with your chosen
    hyperparameters when you call `get_model()`).

HOW?
    This module is not usually imported directly -- use `get_model()` from
    `src.models.factory`, which merges this registry with the classification one and
    handles instantiation + hyperparameters for you.
"""
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

REGRESSION_MODELS = {
    "linear_regression": LinearRegression,
    "linear": LinearRegression,  # alias
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "decision_tree_regressor": DecisionTreeRegressor,
    "random_forest_regressor": RandomForestRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
}
