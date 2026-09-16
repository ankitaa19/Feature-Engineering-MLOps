"""
src/data/splitter.py

WHY?
    Every preprocessing statistic your pipeline learns (an imputer's median, a scaler's mean,
    an encoder's category list) must be learned ONLY from training data. If you split AFTER
    fitting any preprocessing step, your "test" set has already leaked information into that
    fitted statistic, and your evaluation score is no longer a trustworthy estimate of how the
    model will perform on genuinely new data. This is why `split_data()` is one of the very
    first things you call in this framework -- everything downstream depends on it having
    already happened.

WHAT?
    `split_data()` separates a DataFrame into features (X) and target (y), then splits both
    into train and test sets using scikit-learn's `train_test_split`, with the target-aware
    conveniences (stratification for classification) built in.

HOW?
    X_train, X_test, y_train, y_test = split_data(df, target="churn", test_size=0.2, random_state=42)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
):
    """Split a DataFrame into train/test features and targets.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset, features and target together.
    target : str
        Name of the target column.
    test_size : float, default 0.2
        Fraction of rows held out for testing.
    random_state : int, default 42
        Seed for reproducibility -- always set this so your split is repeatable.
    stratify : bool, default False
        If True, preserve the target's class proportions in both splits. Use this for
        classification problems, especially with imbalanced classes. Leave False for
        regression (a continuous target can't be stratified directly).

    Returns
    -------
    X_train, X_test, y_train, y_test

    Raises
    ------
    ValueError
        If `target` is not a column in `df`.
    """
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    logger.info(
        f"Split data: train={X_train.shape}, test={X_test.shape}, "
        f"stratified={stratify}"
    )
    return X_train, X_test, y_train, y_test
