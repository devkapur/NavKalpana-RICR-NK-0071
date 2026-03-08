import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

try:
    from xgboost import XGBClassifier
except ImportError:  
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  
    LGBMClassifier = None

try:
    import shap
except ImportError:  
    shap = None

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    metrics_file_path: str = os.path.join("artifacts", "metrics.json")
    feature_names_file_path: str = os.path.join("artifacts", "feature_names.json")
    shap_summary_file_path: str = os.path.join("artifacts", "shap_summary.json")
    threshold_target_recall: float = 0.85
    threshold_precision_floor: float = 0.60
    ci_bootstrap_iterations: int = 300
    ci_alpha: float = 0.95
    shap_sample_size: int = 200


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _candidate_models(self, y_train: np.ndarray) -> Dict[str, object]:
        positive_count = max(int((y_train == 1).sum()), 1)
        negative_count = max(int((y_train == 0).sum()), 1)
        class_ratio = float(negative_count / positive_count)

        models: Dict[str, object] = {
            "log_reg": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced_subsample"
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "neural_network": Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False)),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
                ]
            ),
        }

        if XGBClassifier is not None:
            models["xgboost"] = XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=class_ratio,
                eval_metric="logloss",
                random_state=42,
            )

        if LGBMClassifier is not None:
            models["lightgbm"] = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
            )

        if {"xgboost", "lightgbm", "neural_network"}.issubset(models.keys()):
            models["ensemble"] = VotingClassifier(
                estimators=[
                    ("xgb", models["xgboost"]),
                    ("lgbm", models["lightgbm"]),
                    ("nn", models["neural_network"]),
                ],
                voting="soft",
            )

        return models

    def _metrics(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        y_pred = (y_prob >= threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "recall": float(recall_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    def _threshold_policy(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_recall: float,
        precision_floor: float,
    ) -> Tuple[float, Dict[str, float], Dict[str, object]]:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        precisions = precisions[:-1]
        recalls = recalls[:-1]

        meets_recall_and_precision = (recalls >= target_recall) & (precisions >= precision_floor)
        if np.any(meets_recall_and_precision):
            candidate_indices = np.where(meets_recall_and_precision)[0]
            best_idx = candidate_indices[np.argmax(precisions[candidate_indices])]
            policy_reason = "max_precision_subject_to_recall_and_precision_floor"
        elif np.any(recalls >= target_recall):
            candidate_indices = np.where(recalls >= target_recall)[0]
            best_idx = candidate_indices[np.argmax(precisions[candidate_indices])]
            policy_reason = "max_precision_subject_to_recall"
        else:
            best_idx = int(np.argmax(recalls))
            policy_reason = "fallback_max_recall"

        tuned_threshold = float(thresholds[best_idx])
        tuned_metrics = self._metrics(y_true, y_prob, threshold=tuned_threshold)

        policy_report = {
            "target_recall": float(target_recall),
            "precision_floor": float(precision_floor),
            "selected_policy": policy_reason,
            "target_recall_achieved": bool(tuned_metrics["recall"] >= target_recall),
            "target_precision_floor_achieved": bool(tuned_metrics["precision"] >= precision_floor),
        }
        return tuned_threshold, tuned_metrics, policy_report

    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric_function: Callable[[np.ndarray, np.ndarray], float],
        iterations: int,
        alpha: float,
    ) -> Dict[str, float]:
        rng = np.random.default_rng(42)
        n = len(y_true)
        values: List[float] = []

        for _ in range(iterations):
            sample_indices = rng.integers(0, n, size=n)
            sample_y_true = y_true[sample_indices]
            sample_y_prob = y_prob[sample_indices]

            if len(np.unique(sample_y_true)) < 2:
                continue

            try:
                values.append(float(metric_function(sample_y_true, sample_y_prob)))
            except ValueError:
                continue

        if not values:
            return {"lower": float("nan"), "upper": float("nan"), "median": float("nan")}

        lower_q = (1.0 - alpha) / 2.0
        upper_q = 1.0 - lower_q
        return {
            "lower": float(np.quantile(values, lower_q)),
            "upper": float(np.quantile(values, upper_q)),
            "median": float(np.median(values)),
        }

    def _calibrate(
        self, best_model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[object, str, Dict[str, float]]:
        calibrated_sigmoid = self._fit_calibrator(best_model, x_train, y_train, method="sigmoid")
        sigmoid_prob = calibrated_sigmoid.predict_proba(x_test)[:, 1]
        sigmoid_metrics = self._metrics(y_test, sigmoid_prob)

        calibrated_isotonic = self._fit_calibrator(best_model, x_train, y_train, method="isotonic")
        isotonic_prob = calibrated_isotonic.predict_proba(x_test)[:, 1]
        isotonic_metrics = self._metrics(y_test, isotonic_prob)

        if isotonic_metrics["roc_auc"] >= sigmoid_metrics["roc_auc"]:
            return calibrated_isotonic, "isotonic", isotonic_metrics
        return calibrated_sigmoid, "sigmoid", sigmoid_metrics

    def _fit_calibrator(
        self,
        fitted_model: object,
        x_train: np.ndarray,
        y_train: np.ndarray,
        method: str,
    ) -> CalibratedClassifierCV:
        try:
            # Backward-compatible path for older sklearn.
            calibrator = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
            calibrator.fit(x_train, y_train)
            return calibrator
        except Exception:
            pass

        if FrozenEstimator is not None:
            # Preferred path for newer sklearn versions where cv='prefit' is removed.
            calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method, cv=None)
            calibrator.fit(x_train, y_train)
            return calibrator

        # Final fallback: re-fit via CV on a cloned model.
        calibrator = CalibratedClassifierCV(clone(fitted_model), method=method, cv=3)
        calibrator.fit(x_train, y_train)
        return calibrator

    def _load_feature_names(self, n_features: int) -> List[str]:
        if os.path.exists(self.model_trainer_config.feature_names_file_path):
            with open(self.model_trainer_config.feature_names_file_path, "r", encoding="utf-8") as feature_file:
                feature_names = json.load(feature_file)
                if isinstance(feature_names, list) and len(feature_names) == n_features:
                    return [str(name) for name in feature_names]
        return [f"feature_{i}" for i in range(n_features)]

    def _save_shap_summary(
        self,
        fitted_model: object,
        x_reference: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, object]:
        if shap is None:
            shap_report = {
                "status": "skipped",
                "reason": "shap_dependency_missing",
                "top_features_by_mean_abs_shap": [],
                "individual_explanations": [],
            }
            with open(self.model_trainer_config.shap_summary_file_path, "w", encoding="utf-8") as shap_file:
                json.dump(shap_report, shap_file, indent=2)
            return shap_report

        sample_size = min(self.model_trainer_config.shap_sample_size, x_reference.shape[0])
        x_sample = x_reference[:sample_size]

        try:
            explainer = shap.Explainer(fitted_model.predict_proba, x_sample)
            shap_result = explainer(x_sample)
            shap_values = shap_result.values
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
        except Exception:
            shap_report = {
                "status": "failed",
                "reason": "shap_computation_error",
                "top_features_by_mean_abs_shap": [],
                "individual_explanations": [],
            }
            with open(self.model_trainer_config.shap_summary_file_path, "w", encoding="utf-8") as shap_file:
                json.dump(shap_report, shap_file, indent=2)
            return shap_report

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:10]

        top_features = [
            {
                "feature": feature_names[int(idx)],
                "mean_abs_shap": float(mean_abs_shap[int(idx)]),
            }
            for idx in top_indices
        ]

        individual_explanations: List[Dict[str, object]] = []
        local_samples = min(5, shap_values.shape[0])
        for row_idx in range(local_samples):
            row_values = shap_values[row_idx]
            local_top_indices = np.argsort(np.abs(row_values))[::-1][:3]
            local_drivers = [
                {
                    "feature": feature_names[int(feature_idx)],
                    "shap_contribution": float(row_values[int(feature_idx)]),
                }
                for feature_idx in local_top_indices
            ]
            individual_explanations.append(
                {
                    "sample_index": int(row_idx),
                    "top_drivers": local_drivers,
                }
            )

        shap_report = {
            "status": "ok",
            "sample_size": int(sample_size),
            "top_features_by_mean_abs_shap": top_features,
            "individual_explanations": individual_explanations,
        }
        with open(self.model_trainer_config.shap_summary_file_path, "w", encoding="utf-8") as shap_file:
            json.dump(shap_report, shap_file, indent=2)
        return shap_report

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> Dict[str, object]:
        try:
            x_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            x_test, y_test = test_array[:, :-1], test_array[:, -1].astype(int)

            models = self._candidate_models(y_train)
            if len(models) < 4:
                raise ValueError("At least 4 candidate models are required for comparison.")

            model_scores: Dict[str, Dict[str, float]] = {}
            fitted_models: Dict[str, object] = {}

            for model_name, model in models.items():
                logging.info("Training model: %s", model_name)
                model.fit(x_train, y_train)
                y_prob = model.predict_proba(x_test)[:, 1]
                model_scores[model_name] = self._metrics(y_test, y_prob)
                fitted_models[model_name] = model

            best_name = max(model_scores.keys(), key=lambda name: model_scores[name]["roc_auc"])
            best_model = fitted_models[best_name]

            calibrated_model, calibration_method, calibrated_metrics = self._calibrate(
                best_model, x_train, y_train, x_test, y_test
            )
            calibrated_prob = calibrated_model.predict_proba(x_test)[:, 1]
            tuned_threshold, tuned_metrics, threshold_policy_report = self._threshold_policy(
                y_test,
                calibrated_prob,
                target_recall=self.model_trainer_config.threshold_target_recall,
                precision_floor=self.model_trainer_config.threshold_precision_floor,
            )

            save_object(self.model_trainer_config.trained_model_file_path, calibrated_model)

            feature_names = self._load_feature_names(n_features=x_test.shape[1])
            shap_report = self._save_shap_summary(
                fitted_model=best_model,
                x_reference=x_test,
                feature_names=feature_names,
            )

            ci_iterations = self.model_trainer_config.ci_bootstrap_iterations
            ci_alpha = self.model_trainer_config.ci_alpha
            confidence_intervals = {
                "roc_auc": self._bootstrap_ci(
                    y_test,
                    calibrated_prob,
                    metric_function=roc_auc_score,
                    iterations=ci_iterations,
                    alpha=ci_alpha,
                ),
                "pr_auc": self._bootstrap_ci(
                    y_test,
                    calibrated_prob,
                    metric_function=average_precision_score,
                    iterations=ci_iterations,
                    alpha=ci_alpha,
                ),
                "recall_at_decision_threshold": self._bootstrap_ci(
                    y_test,
                    calibrated_prob,
                    metric_function=lambda yt, yp: recall_score(yt, (yp >= tuned_threshold).astype(int)),
                    iterations=ci_iterations,
                    alpha=ci_alpha,
                ),
                "precision_at_decision_threshold": self._bootstrap_ci(
                    y_test,
                    calibrated_prob,
                    metric_function=lambda yt, yp: precision_score(
                        yt, (yp >= tuned_threshold).astype(int), zero_division=0
                    ),
                    iterations=ci_iterations,
                    alpha=ci_alpha,
                ),
            }

            final_report = {
                "best_base_model": best_name,
                "calibration_method": calibration_method,
                "base_model_metrics": model_scores,
                "final_calibrated_metrics": calibrated_metrics,
                "decision_threshold": tuned_threshold,
                "threshold_policy": threshold_policy_report,
                "metrics_at_decision_threshold": tuned_metrics,
                "confidence_intervals_95": confidence_intervals,
                "shap_summary_path": self.model_trainer_config.shap_summary_file_path,
                "required_models": {
                    "xgboost_available": bool(XGBClassifier is not None),
                    "lightgbm_available": bool(LGBMClassifier is not None),
                    "neural_network_available": True,
                    "ensemble_available": bool("ensemble" in models),
                },
                "features_used_for_training": feature_names,
            }
            os.makedirs(os.path.dirname(self.model_trainer_config.metrics_file_path), exist_ok=True)
            with open(self.model_trainer_config.metrics_file_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2)

            logging.info("Model training and calibration completed.")
            return final_report
        except Exception as e:
            raise CustomException(e, sys)
