"""Data loading and splitting for the MIT-BIH heartbeat dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
SEED = 42


def load_mitbih(split="train", data_dir=DATA_DIR):
    """Load mitbih_{split}.csv and return signals X (n, 187) and integer labels y (n,)."""
    df = pd.read_csv(Path(data_dir) / f"mitbih_{split}.csv", header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy().astype(np.int64)
    return X, y


def split_train_val(X, y, val_size=0.2, seed=SEED):
    """Stratified train/validation split. The official test set is never used here."""
    return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)
