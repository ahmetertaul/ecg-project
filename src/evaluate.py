"""Evaluation metrics, result saving and plotting helpers."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.data import CLASS_NAMES

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIG_DIR = RESULTS_DIR / "figures"


def compute_metrics(y_true, y_pred):
    """Overall and per-class classification metrics."""
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {
            name: {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
    }


def per_class_table(metrics):
    """Per-class metrics as a DataFrame (one row per class)."""
    return pd.DataFrame(metrics["per_class"]).T


def summary_row(name, metrics):
    """One-line summary used to compare experiments."""
    row = {
        "model": name,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
    }
    for cls in ["S", "V", "F"]:
        row[f"recall_{cls}"] = metrics["per_class"][cls]["recall"]
    return row


def save_results(name, metrics, params=None, history=None, out_dir=METRICS_DIR):
    """Save metrics, experiment settings and (optionally) the training history to results/metrics/{name}.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "params": params or {}, "metrics": metrics}
    if history is not None:
        record["history"] = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(record, f, indent=2, default=float)


def plot_confusion_matrix(y_true, y_pred, title=None, ax=None):
    """Row-normalized confusion matrix: each row shows how the beats of a true class were predicted."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)), normalize="true")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar=False, square=True, linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(axis="y", rotation=0)
    if title:
        ax.set_title(title)
    return ax


def plot_history(history, title=None):
    """Training/validation loss and accuracy, plus validation macro F1, per epoch."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    best_epoch = int(np.argmax(history["val_macro_f1"])) + 1
    train_color, val_color = "#2a78d6", "#eb6834"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("loss", "Loss"), ("accuracy", "Accuracy")]
    for ax, (key, label) in zip(axes, panels):
        ax.plot(epochs, history[key], color=train_color, linewidth=2, label="Train")
        ax.plot(epochs, history[f"val_{key}"], color=val_color, linewidth=2, label="Validation")
        ax.set_ylabel(label)
        ax.legend(frameon=False)
    axes[2].plot(epochs, history["val_macro_f1"], color=val_color, linewidth=2)
    axes[2].set_ylabel("Validation macro F1")

    for ax in axes:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].annotate(f"best epoch = {best_epoch}", xy=(best_epoch, axes[2].get_ylim()[0]),
                     xytext=(4, 6), textcoords="offset points", fontsize=9, color="gray")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
