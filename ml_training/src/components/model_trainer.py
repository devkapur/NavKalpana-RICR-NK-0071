import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
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


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    metrics_file_path: str = os.path.join("artifacts", "metrics.json")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _candidate_models(self) -> Dict[str, object]:
        models: Dict[str, object] = {
            "log_reg": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced_subsample"
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "mlp": Pipeline(
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
                eval_metric="logloss",
                random_state=42,
            )

        if "xgboost" in models:
            models["ensemble"] = VotingClassifier(
                estimators=[
                    ("xgb", models["xgboost"]),
                    ("rf", models["random_forest"]),
                    ("lr", models["log_reg"]),
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
        }

    def _threshold_for_target_recall(
        self, y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.85
    ) -> Tuple[float, Dict[str, float], bool]:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        precisions = precisions[:-1]
        recalls = recalls[:-1]

        meets_target = recalls >= target_recall
        if np.any(meets_target):
            candidate_indices = np.where(meets_target)[0]
            best_idx = candidate_indices[np.argmax(precisions[candidate_indices])]
            achieved_target = True
        else:
            best_idx = int(np.argmax(recalls))
            achieved_target = False

        tuned_threshold = float(thresholds[best_idx])
        tuned_metrics = self._metrics(y_true, y_prob, threshold=tuned_threshold)
        return tuned_threshold, tuned_metrics, achieved_target

    def _calibrate(
        self, best_model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[object, str, Dict[str, float]]:
        calibrated_sigmoid = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
        calibrated_sigmoid.fit(x_train, y_train)
        sigmoid_prob = calibrated_sigmoid.predict_proba(x_test)[:, 1]
        sigmoid_metrics = self._metrics(y_test, sigmoid_prob)

        calibrated_isotonic = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
        calibrated_isotonic.fit(x_train, y_train)
        isotonic_prob = calibrated_isotonic.predict_proba(x_test)[:, 1]
        isotonic_metrics = self._metrics(y_test, isotonic_prob)

        if isotonic_metrics["roc_auc"] >= sigmoid_metrics["roc_auc"]:
            return calibrated_isotonic, "isotonic", isotonic_metrics
        return calibrated_sigmoid, "sigmoid", sigmoid_metrics

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> Dict[str, object]:
        try:
            x_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            x_test, y_test = test_array[:, :-1], test_array[:, -1].astype(int)

            models = self._candidate_models()
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
            tuned_threshold, tuned_metrics, target_recall_achieved = self._threshold_for_target_recall(
                y_test, calibrated_prob, target_recall=0.85
            )

            save_object(self.model_trainer_config.trained_model_file_path, calibrated_model)

            final_report = {
                "best_base_model": best_name,
                "calibration_method": calibration_method,
                "base_model_metrics": model_scores,
                "final_calibrated_metrics": calibrated_metrics,
                "decision_threshold": tuned_threshold,
                "threshold_target_recall": 0.85,
                "target_recall_achieved": target_recall_achieved,
                "metrics_at_decision_threshold": tuned_metrics,
            }
            os.makedirs(os.path.dirname(self.model_trainer_config.metrics_file_path), exist_ok=True)
            with open(self.model_trainer_config.metrics_file_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2)

            logging.info("Model training and calibration completed.")
            return final_report
        except Exception as e:
            raise CustomException(e, sys)
