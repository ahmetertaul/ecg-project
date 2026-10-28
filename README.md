# ECG Heartbeat Arrhythmia Classification

Deep learning-based classification of individual ECG heartbeats into the five AAMI classes
(N, S, V, F, Q) using the MIT-BIH Arrhythmia dataset.

LEEN364 – Deep Learning and Classification Techniques, Istanbul Arel University.

## Results

Test set (official MIT-BIH split, 21,892 beats), mean ± std over three seeds (42, 123, 2026):

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| Logistic regression | 0.915 ± 0.000 | 0.597 ± 0.000 | 0.664 ± 0.000 |
| Random forest (balanced) | 0.977 ± 0.000 | 0.865 ± 0.002 | 0.890 ± 0.001 |
| MLP | 0.980 ± 0.000 | 0.857 ± 0.006 | 0.893 ± 0.004 |
| Simple 1D CNN | 0.985 ± 0.000 | 0.889 ± 0.001 | 0.915 ± 0.004 |
| Residual 1D CNN | 0.985 ± 0.004 | 0.904 ± 0.016 | 0.918 ± 0.015 |
| **Residual 1D CNN + oversampling + augmentation** | **0.986 ± 0.001** | **0.935 ± 0.001** | **0.925 ± 0.004** |
| LSTM (2 of 3 runs, one run did not converge) | 0.976 ± 0.002 | 0.836 ± 0.016 | 0.875 ± 0.007 |

Macro F1-score is the main metric because of the strong class imbalance (83% of the beats are normal).
All model choices were made on a stratified validation split (20% of the training file);
the test set was used only once for the final evaluation.

## Dataset

[ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
(Kachuee et al., 2018). Download `mitbih_train.csv` and `mitbih_test.csv` and place them in `data/raw/`.

## Setup

Python 3.13 was used. TensorFlow runs on the CPU on native Windows.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Running the experiments

Run the notebooks in order from the `notebooks/` folder:

1. `01_eda.ipynb` – class distribution, example beats, data quality checks, train/validation split
2. `02_baselines.ipynb` – logistic regression and random forest
3. `03_deep_models.ipynb` – MLP, simple 1D CNN, residual 1D CNN and LSTM (seed 42)
4. `04_imbalance.ipynb` – class weights, oversampling and data augmentation on the residual CNN

Then train the final models with three seeds (about 2 hours on a CPU, most of it for the LSTM).
Seed-42 models from notebooks 03 and 04 are reused, and finished runs are skipped if the script is restarted:

```bash
python -m src.run_final
```

5. `05_final_evaluation.ipynb` – single evaluation on the test set, final tables and figures

Trained models are saved to `results/models/` (not tracked by git).

## Project structure

```
data/raw/            raw CSV files (not tracked)
notebooks/           EDA, baselines, deep models, imbalance experiments, final evaluation
src/
  data.py            data loading and train/validation split
  models.py          MLP, CNN, residual CNN and LSTM definitions
  train.py           training loop (early stopping on validation macro F1)
  augment.py         data augmentation for training batches
  evaluate.py        metrics, result saving and plots
  run_final.py       training of the final models with three seeds
results/
  figures/           all figures (PNG, 200 dpi)
  metrics/           metrics and settings of every experiment (JSON / CSV)
report/              project report
```
