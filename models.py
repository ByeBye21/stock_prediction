"""
models.py - Classification-only model training, calibration, and evaluation

Models:
  1. Logistic Regression - simple linear benchmark
  2. Random Forest Classifier - ensemble tree model
  3. XGBoost Classifier - gradient boosting
  4. LightGBM Classifier - fast gradient boosting

Evaluation:
  - Walk-forward validation via TimeSeriesSplit
  - StandardScaler for feature normalization
  - RandomizedSearchCV for optional hyperparameter tuning
  - Probability calibration on recent in-fold data
  - Decision-threshold tuning for directional calls
  - Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1, MCC, AUC-ROC
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


BINARY_LABELS = [0, 1]


PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [6, 10, 14, None],
        "model__min_samples_split": [4, 8, 12],
        "model__min_samples_leaf": [2, 4, 8],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
    },
    "LightGBM": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8, -1],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__num_leaves": [15, 31, 63],
    },
}


class CalibratedThresholdClassifier:
    """Wrap a fitted classifier with probability calibration and a custom threshold."""

    def __init__(self, base_model, calibrator=None, threshold: float = 0.5):
        self.base_model = base_model
        self.calibrator = calibrator
        self.threshold = float(np.clip(threshold, 0.01, 0.99))

    @property
    def named_steps(self):
        return getattr(self.base_model, "named_steps", {})

    @property
    def classes_(self):
        return np.asarray(BINARY_LABELS)

    def predict_proba(self, X):
        raw_prob = _extract_positive_class_probability(self.base_model, X)
        if raw_prob is None:
            predictions = np.asarray(self.base_model.predict(X), dtype=float)
            raw_prob = np.clip(predictions, 0.0, 1.0)

        calibrated_prob = _apply_probability_calibrator(raw_prob, self.calibrator)
        calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated_prob, calibrated_prob])

    def predict(self, X):
        prob_up = self.predict_proba(X)[:, 1]
        return _predict_from_threshold(prob_up, self.threshold)


def _safe_macro_scores(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> tuple[float, float, float]:
    """Compute macro scores across both binary classes, even in one-class folds."""
    precision = precision_score(
        y_true,
        y_pred,
        labels=BINARY_LABELS,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        labels=BINARY_LABELS,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        labels=BINARY_LABELS,
        average="macro",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def _safe_auc_binary(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None,
) -> float:
    """Return a neutral AUC when the evaluation window lacks both classes."""
    unique_true = pd.Series(y_true).dropna().unique()
    if y_prob is None or len(unique_true) < 2:
        return 0.5

    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def _safe_mcc(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Compute MCC safely when only one class is present."""
    combined_labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    if len(y_true) == 0 or len(combined_labels) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def get_models(selected: list[str]) -> dict:
    """Return classifier pipelines for the selected model names."""
    available = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss",
            )),
        ]),
        "LightGBM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )),
        ]),
    }
    return {name: available[name] for name in selected if name in available}


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute classification metrics for stock-direction prediction."""
    acc = accuracy_score(y_true, y_pred) * 100
    prec, rec, f1 = _safe_macro_scores(y_true, y_pred)
    balanced_acc = rec * 100
    prec *= 100
    rec *= 100
    f1 *= 100
    mcc = _safe_mcc(y_true, y_pred)
    auc = _safe_auc_binary(y_true, y_prob) * 100

    return {
        "Accuracy (%)": float(round(acc, 2)),
        "Balanced Accuracy (%)": float(round(balanced_acc, 2)),
        "Macro Precision (%)": float(round(prec, 2)),
        "Macro Recall (%)": float(round(rec, 2)),
        "Macro F1-Score (%)": float(round(f1, 2)),
        "MCC": float(round(mcc, 4)),
        "AUC-ROC (%)": float(round(auc, 2)),
    }


def get_model_selection_score(metrics: dict) -> tuple:
    """Return a sortable score tuple for model selection in the UI."""
    return (
        float(metrics.get("AUC-ROC (%)", 50.0)),
        float(metrics.get("Balanced Accuracy (%)", 50.0)),
        float(metrics.get("MCC", 0.0)),
        float(metrics.get("Accuracy (%)", 0.0)),
        float(metrics.get("Macro F1-Score (%)", 0.0)),
    )


def get_best_model_name(results: dict) -> str:
    """Pick the best model using classification-focused ranking criteria."""
    if not results:
        raise ValueError("No model results available.")

    return max(
        results,
        key=lambda name: get_model_selection_score(results[name]["metrics"]),
    )


def _resolve_n_splits(n_samples: int, requested_splits: int, horizon: int) -> int:
    """Find the largest valid split count for the sample size and horizon."""
    gap = max(int(horizon), 0)
    for splits in range(requested_splits, 1, -1):
        try:
            list(TimeSeriesSplit(n_splits=splits, gap=gap).split(range(n_samples)))
            return splits
        except ValueError:
            continue

    raise ValueError(
        "Not enough data for walk-forward validation with the selected horizon. "
        "Please choose a wider date range or a shorter target horizon."
    )


def _get_time_series_split(
    n_samples: int,
    n_splits: int,
    horizon: int,
) -> TimeSeriesSplit:
    """Create a leakage-safe time series splitter for forward-direction targets."""
    resolved_splits = _resolve_n_splits(n_samples, n_splits, horizon)
    return TimeSeriesSplit(n_splits=resolved_splits, gap=max(int(horizon), 0))


def _get_class_values(y: pd.Series | np.ndarray) -> np.ndarray:
    """Return the distinct non-null class labels present in a target series."""
    values = pd.Series(y).dropna().to_numpy()
    return np.unique(values)


def _has_multiple_classes(y: pd.Series | np.ndarray) -> bool:
    """Whether a classification target contains at least two classes."""
    return len(_get_class_values(y)) >= 2


def _build_single_class_classifier(y: pd.Series | np.ndarray) -> Pipeline:
    """Fallback classifier for windows that only contain one observed class."""
    classes = _get_class_values(y)
    constant = int(classes[0]) if len(classes) else 1
    return Pipeline([
        ("model", DummyClassifier(strategy="constant", constant=constant)),
    ])


def _get_model_classes(model) -> np.ndarray | None:
    """Read class labels from a model or pipeline when available."""
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes)

    named_steps = getattr(model, "named_steps", None)
    if named_steps is None:
        return None

    final_model = named_steps.get("model")
    if final_model is None:
        return None

    classes = getattr(final_model, "classes_", None)
    return None if classes is None else np.asarray(classes)


def _extract_positive_class_probability(model, X) -> np.ndarray | None:
    """Return P(class=1) for classifiers, including one-class fallback models."""
    if not hasattr(model, "predict_proba"):
        return None

    try:
        prob = np.asarray(model.predict_proba(X), dtype=float)
    except Exception:
        return None

    if prob.ndim != 2 or prob.shape[0] != len(X) or prob.shape[1] == 0:
        return None

    classes = _get_model_classes(model)
    if prob.shape[1] == 1:
        if classes is not None and len(classes) == 1:
            return np.full(len(X), 1.0 if int(classes[0]) == 1 else 0.0, dtype=float)
        return prob[:, 0]

    if classes is not None and 1 in classes:
        positive_idx = int(np.where(classes == 1)[0][0])
        return prob[:, positive_idx]

    return prob[:, -1]


def _fit_probability_calibrator(
    y_prob: np.ndarray | pd.Series | None,
    y_true: np.ndarray | pd.Series,
):
    """Fit a simple Platt-style calibrator on held-out probabilities."""
    if y_prob is None:
        return None

    prob_array = np.asarray(y_prob, dtype=float).reshape(-1, 1)
    truth_array = np.asarray(y_true, dtype=int)

    if len(prob_array) != len(truth_array) or len(truth_array) < 30:
        return None
    if len(np.unique(truth_array)) < 2 or len(np.unique(prob_array)) < 2:
        return None

    calibrator = LogisticRegression(max_iter=1000, random_state=42)
    try:
        calibrator.fit(prob_array, truth_array)
        return calibrator
    except Exception:
        return None


def _apply_probability_calibrator(
    y_prob: np.ndarray | pd.Series | None,
    calibrator,
) -> np.ndarray | None:
    """Apply a fitted probability calibrator when available."""
    if y_prob is None:
        return None

    prob_array = np.asarray(y_prob, dtype=float)
    if calibrator is None:
        return np.clip(prob_array, 0.0, 1.0)

    try:
        calibrated = calibrator.predict_proba(prob_array.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
    except Exception:
        return np.clip(prob_array, 0.0, 1.0)


def _predict_from_threshold(
    y_prob: np.ndarray | pd.Series | None,
    threshold: float,
) -> np.ndarray:
    """Convert positive-class probabilities into hard labels."""
    if y_prob is None:
        return np.array([], dtype=int)
    return (np.asarray(y_prob, dtype=float) >= float(threshold)).astype(int)


def _tune_decision_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None,
) -> float:
    """Choose a stable classification threshold using recent held-out data."""
    if y_prob is None or len(np.asarray(y_true)) == 0 or len(np.unique(y_true)) < 2:
        return 0.5

    best_threshold = 0.5
    best_score = None
    truth_array = np.asarray(y_true, dtype=int)
    prob_array = np.asarray(y_prob, dtype=float)

    for threshold in np.linspace(0.30, 0.70, 41):
        pred = _predict_from_threshold(prob_array, threshold)
        _, macro_recall, macro_f1 = _safe_macro_scores(truth_array, pred)
        score = (
            macro_recall,
            macro_f1,
            accuracy_score(truth_array, pred),
            -abs(float(threshold) - 0.5),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def _should_apply_probability_calibration(model_name: str) -> bool:
    """
    Decide whether a model should receive the extra Platt-style calibration layer.

    Logistic Regression already optimizes log-loss directly and usually emits
    usable probabilities, so a second calibration step often adds noise instead
    of improving the ranking quality shown in the app.
    """
    return model_name != "Logistic Regression"


def _get_recent_calibration_split(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    calibration_fraction: float = 0.20,
    min_calibration_size: int = 40,
):
    """Reserve the most recent in-fold window for calibration and threshold tuning."""
    n_samples = len(X)
    min_fit_size = max(80, 4 * max(int(horizon), 1))
    max_calibration_size = n_samples - min_fit_size

    if max_calibration_size < min_calibration_size:
        return X, y, None, None

    target_size = min(
        max(int(n_samples * calibration_fraction), min_calibration_size),
        max_calibration_size,
    )

    for calibration_size in range(target_size, min_calibration_size - 1, -5):
        split_idx = n_samples - calibration_size
        X_fit = X.iloc[:split_idx]
        y_fit = y.iloc[:split_idx]
        X_cal = X.iloc[split_idx:]
        y_cal = y.iloc[split_idx:]

        if _has_multiple_classes(y_fit) and _has_multiple_classes(y_cal):
            return X_fit, y_fit, X_cal, y_cal

    return X, y, None, None


def _tuning_score(estimator, X, y) -> float:
    """
    Safer tuning score for time-series classification.

    AUC remains the primary signal, while macro recall keeps one-class windows
    from looking artificially perfect.
    """
    y_pred = estimator.predict(X)
    y_prob = _extract_positive_class_probability(estimator, X)
    _, macro_recall, _ = _safe_macro_scores(y, y_pred)
    auc = _safe_auc_binary(y, y_prob)
    return (auc * 100.0) + macro_recall


def fit_final_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    do_tuning: bool = True,
    horizon: int = 1,
) -> Pipeline:
    """Fit the final forecast model on the full dataset after evaluation."""
    if not _has_multiple_classes(y):
        fallback = _build_single_class_classifier(y)
        fallback.fit(X, y)
        return fallback

    try:
        if do_tuning:
            return tune_model(
                pipeline,
                X,
                y,
                model_name,
                horizon=horizon,
            )

        pipeline.fit(X, y)
        return pipeline
    except ValueError:
        fallback = _build_single_class_classifier(y)
        fallback.fit(X, y)
        return fallback


def tune_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    n_iter: int = 15,
    horizon: int = 1,
) -> Pipeline:
    """
    Hyperparameter tuning via RandomizedSearchCV.
    Only runs for models with a defined parameter grid.
    """
    if model_name not in PARAM_GRIDS:
        pipeline.fit(X_train, y_train)
        return pipeline

    if not _has_multiple_classes(y_train):
        fallback = _build_single_class_classifier(y_train)
        fallback.fit(X_train, y_train)
        return fallback

    try:
        search_cv = _get_time_series_split(
            n_samples=len(X_train),
            n_splits=3,
            horizon=horizon,
        )
    except ValueError:
        pipeline.fit(X_train, y_train)
        return pipeline

    param_grid = PARAM_GRIDS[model_name]
    search_n_iter = min(n_iter, len(ParameterGrid(param_grid)))

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=search_n_iter,
        cv=search_cv,
        scoring=_tuning_score,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        error_score=np.nan,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FitFailedWarning)
            warnings.simplefilter("ignore", UserWarning)
            search.fit(X_train, y_train)

        mean_scores = np.asarray(search.cv_results_.get("mean_test_score", []), dtype=float)
        if mean_scores.size == 0 or not np.isfinite(mean_scores).any():
            pipeline.fit(X_train, y_train)
            return pipeline
        return search.best_estimator_
    except ValueError:
        try:
            pipeline.fit(X_train, y_train)
            return pipeline
        except ValueError:
            fallback = _build_single_class_classifier(y_train)
            fallback.fit(X_train, y_train)
            return fallback


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    selected_models: list[str],
    n_splits: int = 5,
    do_tuning: bool = True,
    horizon: int = 1,
    progress_callback=None,
) -> dict:
    """
    Train and evaluate selected classifiers using walk-forward validation.

    Args:
        X: Feature DataFrame
        y: Target Series
        selected_models: List of model names
        n_splits: Number of CV folds
        do_tuning: Whether to run hyperparameter tuning
        horizon: Prediction horizon used for leakage-safe CV gaps
        progress_callback: Progress notification function

    Returns:
        dict: Results for each model
    """
    models = get_models(selected_models)
    results = {}
    tscv = _get_time_series_split(n_samples=len(X), n_splits=n_splits, horizon=horizon)
    total_models = len(models)

    for model_idx, (model_name, pipeline) in enumerate(models.items()):
        fold_metrics = []
        fold_thresholds = []
        base_pipeline = clone(pipeline)

        if progress_callback:
            progress_callback(model_idx, total_models, f"Training {model_name}...")

        all_y_pred = []
        all_y_test = []
        all_dates = []
        all_y_prob = []
        all_raw_y_prob = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            X_fit, y_fit, X_cal, y_cal = _get_recent_calibration_split(
                X_train,
                y_train,
                horizon=horizon,
            )
            fold_pipeline = clone(base_pipeline)

            try:
                if not _has_multiple_classes(y_fit):
                    fold_pipeline = _build_single_class_classifier(y_fit)
                    fold_pipeline.fit(X_fit, y_fit)
                elif do_tuning:
                    fold_pipeline = tune_model(
                        fold_pipeline,
                        X_fit,
                        y_fit,
                        model_name,
                        horizon=horizon,
                    )
                else:
                    fold_pipeline.fit(X_fit, y_fit)
            except ValueError:
                fold_pipeline = _build_single_class_classifier(y_fit)
                fold_pipeline.fit(X_fit, y_fit)

            raw_test_prob = _extract_positive_class_probability(fold_pipeline, X_test)
            if raw_test_prob is None:
                raw_test_prob = np.asarray(fold_pipeline.predict(X_test), dtype=float)

            calibrator = None
            tuned_threshold = 0.5
            calibrated_test_prob = np.clip(np.asarray(raw_test_prob, dtype=float), 0.0, 1.0)

            if X_cal is not None and y_cal is not None:
                raw_cal_prob = _extract_positive_class_probability(fold_pipeline, X_cal)
                if raw_cal_prob is not None:
                    calibration_prob = np.clip(np.asarray(raw_cal_prob, dtype=float), 0.0, 1.0)
                    if _should_apply_probability_calibration(model_name):
                        calibrator = _fit_probability_calibrator(raw_cal_prob, y_cal)
                        calibration_prob = _apply_probability_calibrator(raw_cal_prob, calibrator)
                        calibrated_test_prob = _apply_probability_calibrator(raw_test_prob, calibrator)
                    tuned_threshold = _tune_decision_threshold(y_cal.values, calibration_prob)

            y_pred = _predict_from_threshold(calibrated_test_prob, tuned_threshold)
            metrics = calculate_classification_metrics(
                y_test.values,
                y_pred,
                y_prob=calibrated_test_prob,
            )
            fold_metrics.append(metrics)
            fold_thresholds.append(tuned_threshold)

            all_y_pred.extend(y_pred)
            all_y_test.extend(y_test.values)
            all_dates.extend(X.iloc[test_idx].index)
            all_y_prob.extend(calibrated_test_prob)
            all_raw_y_prob.extend(np.asarray(raw_test_prob, dtype=float))

        final_base_model = fit_final_model(
            clone(base_pipeline),
            X,
            y,
            model_name,
            do_tuning=do_tuning,
            horizon=horizon,
        )

        overall_y_test = np.asarray(all_y_test, dtype=int)
        overall_y_pred = np.asarray(all_y_pred, dtype=int)
        overall_y_prob = (
            np.asarray(all_y_prob, dtype=float)
            if len(all_y_prob) == len(all_y_test) and len(all_y_prob) > 0
            else None
        )
        overall_metrics = calculate_classification_metrics(
            overall_y_test,
            overall_y_pred,
            y_prob=overall_y_prob,
        )

        decision_threshold = (
            float(np.median(fold_thresholds))
            if fold_thresholds
            else 0.5
        )
        global_calibrator = None
        if _should_apply_probability_calibration(model_name):
            global_calibrator = _fit_probability_calibrator(all_raw_y_prob, overall_y_test)
        final_model = CalibratedThresholdClassifier(
            final_base_model,
            calibrator=global_calibrator,
            threshold=decision_threshold,
        )

        overall_metrics["Decision Threshold (%)"] = float(round(decision_threshold * 100, 2))

        importance = None
        actual_model = final_model.named_steps.get("model", None)
        if actual_model is not None and hasattr(actual_model, "feature_importances_"):
            importance = pd.Series(
                actual_model.feature_importances_,
                index=X.columns,
            ).sort_values(ascending=False)
        elif actual_model is not None and hasattr(actual_model, "coef_"):
            coef = np.asarray(actual_model.coef_, dtype=float)
            if coef.ndim == 1:
                importance_values = np.abs(coef)
            else:
                importance_values = np.abs(coef).mean(axis=0)
            importance = pd.Series(
                importance_values,
                index=X.columns,
            ).sort_values(ascending=False)

        results[model_name] = {
            "model": final_model,
            "metrics": overall_metrics,
            "fold_metrics": fold_metrics,
            "y_test": pd.Series(all_y_test, index=all_dates),
            "y_pred": np.array(all_y_pred),
            "y_prob": overall_y_prob,
            "test_dates": pd.DatetimeIndex(all_dates),
            "feature_importance": importance,
            "decision_threshold": decision_threshold,
            "is_calibrated": global_calibrator is not None,
        }

    if progress_callback:
        progress_callback(total_models, total_models, "All models trained!")

    return results
