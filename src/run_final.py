"""Train the final models with several seeds and save them for the test-set evaluation.

Run from the project root:  python -m src.run_final

Each (model, seed) run is skipped if its result file already exists, so the script can be restarted.
Only the training and validation splits are used here; the test set is evaluated in notebook 05.
"""

import json
import os
import shutil
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_mitbih, split_train_val
from src.evaluate import RESULTS_DIR, METRICS_DIR, compute_metrics, save_results
from src.models import build_lstm, build_mlp, build_resnet_cnn, build_simple_cnn
from src.train import set_seed, train_model

SEEDS = [42, 123, 2026]
FINAL_METRICS_DIR = METRICS_DIR / "final"
FINAL_MODEL_DIR = RESULTS_DIR / "models" / "final"

# Seed-42 runs from notebooks 03 and 04 (same data, settings and seed), reused instead of retrained.
DEV_RUNS = {
    "mlp": "mlp",
    "simple_cnn": "simple_cnn",
    "resnet_cnn": "resnet_cnn",
    "resnet_cnn_D": "exp_D_oversampling_aug",
    "lstm": "lstm",
}

KERAS_MODELS = {
    "mlp": build_mlp,
    "simple_cnn": build_simple_cnn,
    "resnet_cnn": build_resnet_cnn,
    "resnet_cnn_D": build_resnet_cnn,
    "lstm": build_lstm,
}

# Fastest first, LSTM last.
RUN_ORDER = ["logreg", "rf_balanced", "mlp", "simple_cnn", "resnet_cnn", "resnet_cnn_D", "lstm"]


def run_sklearn(name, seed, X_tr, y_tr, X_val, y_val):
    if name == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        params = {"model": "LogisticRegression", "scaler": "StandardScaler", "class_weight": None}
    else:
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed)
        params = {"model": "RandomForestClassifier", "n_estimators": 200, "class_weight": "balanced"}

    start = time.time()
    model.fit(X_tr, y_tr)
    params.update(seed=seed, train_time_s=round(time.time() - start, 1))
    joblib.dump(model, FINAL_MODEL_DIR / f"{name}_seed{seed}.joblib", compress=3)

    metrics = compute_metrics(y_val, model.predict(X_val))
    save_results(f"{name}_seed{seed}", metrics, params, out_dir=FINAL_METRICS_DIR)
    return metrics


def run_keras(name, seed, X_tr, y_tr, X_val, y_val):
    model_path = FINAL_MODEL_DIR / f"{name}_seed{seed}.keras"

    if seed == 42 and name in DEV_RUNS:
        dev = DEV_RUNS[name]
        with open(METRICS_DIR / f"{dev}.json") as f:
            record = json.load(f)
        shutil.copy(RESULTS_DIR / "models" / f"{dev}.keras", model_path)
        params = {**record["params"], "reused_from": dev}
        save_results(f"{name}_seed{seed}", record["metrics"], params, record["history"], out_dir=FINAL_METRICS_DIR)
        return record["metrics"]

    augment = None
    if name == "resnet_cnn_D":
        X_tr, y_tr = RandomOverSampler(random_state=seed).fit_resample(X_tr, y_tr)
        augment = {"seed": seed, "p": 0.5, "noise_std": 0.02, "scale": (0.9, 1.1), "max_shift": 5}

    set_seed(seed)
    model = KERAS_MODELS[name]()
    history, train_time = train_model(model, X_tr, y_tr, X_val, y_val, augment=augment)
    model.save(model_path)

    metrics = compute_metrics(y_val, model.predict(X_val, batch_size=1024, verbose=0).argmax(axis=1))
    params = {"model": name, "seed": seed, "lr": 1e-3, "batch_size": 128, "max_epochs": 50,
              "epochs_trained": len(history.history["loss"]), "train_time_s": round(train_time, 1)}
    if augment is not None:
        params.update(imbalance="random oversampling (all classes to N count)", augmentation=augment)
    save_results(f"{name}_seed{seed}", metrics, params, history.history, out_dir=FINAL_METRICS_DIR)
    return metrics


def main():
    FINAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_mitbih("train")
    X_tr, X_val, y_tr, y_val = split_train_val(X_train, y_train)

    for name in RUN_ORDER:
        for seed in SEEDS:
            if (FINAL_METRICS_DIR / f"{name}_seed{seed}.json").exists():
                print(f"[skip] {name} seed {seed}", flush=True)
                continue
            print(f"[start] {name} seed {seed}", flush=True)
            start = time.time()
            runner = run_keras if name in KERAS_MODELS else run_sklearn
            metrics = runner(name, seed, X_tr, y_tr, X_val, y_val)
            print(f"[done] {name} seed {seed}: val macro F1 {metrics['macro_f1']:.4f} "
                  f"({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
