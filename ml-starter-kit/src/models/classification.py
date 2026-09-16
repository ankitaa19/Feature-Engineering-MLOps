"""
src/models/classification.py

WHY?
    Same idea as the regression registry: swap classifiers by changing one string,
    instead of hunting down and importing a new class every time you want to try
    something different.

WHAT?
    A dictionary (`CLASSIFICATION_MODELS`) mapping short, memorable names to scikit-learn
    classifier CLASSES.

HOW?
    Use `get_model()` from `src.models.factory` -- it merges this registry with the
    regression one and handles instantiation + hyperparameters for you.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

CLASSIFICATION_MODELS = {
    "logistic": LogisticRegression,
    "logistic_regression": LogisticRegression,  # alias
    "decision_tree_classifier": DecisionTreeClassifier,
    "random_forest_classifier": RandomForestClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
    "knn": KNeighborsClassifier,
    "svm": SVC,
}
