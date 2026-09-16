"""
src/utils/visualization.py

WHY?
    A handful of plots get made in nearly every ML project -- a confusion matrix, an ROC
    curve, a feature importance bar chart, a residuals plot. Rewriting matplotlib
    boilerplate for each one, every time, is exactly the kind of repetition this
    framework exists to remove.

WHAT?
    Small, focused plotting functions, each producing one standard chart from
    already-computed results (predictions, importances, etc.) -- not from a raw pipeline,
    so they stay simple and reusable regardless of what produced the numbers.

HOW?
    from src.utils.visualization import plot_confusion_matrix, plot_feature_importance
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(importance_df, top_n=15)
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Plot a confusion matrix as a labeled heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    n = cm.shape[0]
    tick_labels = labels if labels is not None else range(n)
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """Plot an ROC curve with the AUC in the legend."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#028090")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df, top_n: int = 15, title="Feature Importance"):
    """Plot a horizontal bar chart from a feature-importance DataFrame.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must have a "feature" column and either an "importance" or "importance_mean"
        column -- exactly the shape returned by `src.evaluation.importance` functions.
    top_n : int, default 15
        Number of top features to display.
    """
    value_col = "importance" if "importance" in importance_df.columns else "importance_mean"
    plot_df = importance_df.head(top_n).sort_values(value_col)

    fig, ax = plt.subplots(figsize=(7, max(3, top_n * 0.35)))
    ax.barh(plot_df["feature"], plot_df[value_col], color="#028090")
    ax.set_xlabel(value_col)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title="Residuals Plot"):
    """Plot predicted values against residuals (y_true - y_pred) -- a quick way to spot
    patterns a regression model is missing (residuals should look like random noise
    centered on zero, with no obvious shape)."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(y_pred, residuals, alpha=0.5, color="#028090", s=18)
    ax.axhline(0, color="#B5451B", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(title)
    plt.tight_layout()
    return fig
