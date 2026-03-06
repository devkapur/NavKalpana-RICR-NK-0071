import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.utils import load_object

try:
    import numpy as np
    import shap
except ImportError:
    np = None
    shap = None


@dataclass
class PredictPipelineConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    metrics_path: str = os.path.join("artifacts", "metrics.json")
    feature_names_path: str = os.path.join("artifacts", "feature_names.json")
    shap_summary_path: str = os.path.join("artifacts", "shap_summary.json")

class PredictPipeline:
    def __init__(self):
        self.config = PredictPipelineConfig()

    @staticmethod
    def _risk_category(probability: float) -> str:
        if probability < 0.20:
            return "Low Risk"
        if probability < 0.50:
            return "Moderate Risk"
        return "High Risk"

    @staticmethod
    def _recommendations(risk_category: str) -> Dict[str, object]:
        if risk_category == "High Risk":
            return {
                "urgency": "high",
                "summary": "Early clinical evaluation is strongly advised.",
                "actions": [
                    "Consult a physician/cardiologist within 1-2 weeks.",
                    "Repeat blood pressure readings over multiple days.",
                    "Get fasting glucose and lipid profile if pending.",
                    "Start structured lifestyle intervention immediately.",
                ],
            }
        if risk_category == "Moderate Risk":
            return {
                "urgency": "medium",
                "summary": "Preventive follow-up is recommended.",
                "actions": [
                    "Schedule a routine clinical review.",
                    "Track blood pressure and weight weekly.",
                    "Reduce salt/sugar intake and increase physical activity.",
                    "Repeat risk screening in 3-6 months.",
                ],
            }
        return {
            "urgency": "low",
            "summary": "Current risk appears lower; maintain prevention habits.",
            "actions": [
                "Continue healthy diet and regular exercise.",
                "Avoid smoking/alcohol excess.",
                "Monitor blood pressure periodically.",
                "Repeat risk screening in 6-12 months.",
            ],
        }

    def _load_feature_names(self, transformed_width: int) -> List[str]:
        if os.path.exists(self.config.feature_names_path):
            with open(self.config.feature_names_path, "r", encoding="utf-8") as feature_file:
                names = json.load(feature_file)
                if isinstance(names, list) and len(names) == transformed_width:
                    return [str(name) for name in names]
        return [f"feature_{i}" for i in range(transformed_width)]

    def _unwrap_base_estimator(self, model: object) -> object:
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            calibrated = model.calibrated_classifiers_[0]
            if hasattr(calibrated, "estimator"):
                return calibrated.estimator
            if hasattr(calibrated, "base_estimator"):
                return calibrated.base_estimator
        return model

    @staticmethod
    def _friendly_feature_label(feature_name: str) -> str:
        direct_map = {
            "num__age": "Age (days)",
            "num__age_years": "Age (years)",
            "num__height": "Height (cm)",
            "num__weight": "Weight (kg)",
            "num__ap_hi": "Systolic BP",
            "num__ap_lo": "Diastolic BP",
            "num__bmi": "BMI",
            "num__pulse_pressure": "Pulse Pressure",
            "num__age_bp_interaction": "Age x Systolic BP",
            "num__glucose_bmi_interaction": "Glucose x BMI",
        }
        if feature_name in direct_map:
            return direct_map[feature_name]

        if feature_name.startswith("cat__"):
            raw = feature_name.replace("cat__", "", 1)
            parts = raw.split("_")
            if len(parts) >= 2:
                field = parts[0]
                value = "_".join(parts[1:])
                field_map = {
                    "gender": "Gender",
                    "cholesterol": "Cholesterol Level",
                    "gluc": "Glucose Level",
                    "smoke": "Smoking Status",
                    "alco": "Alcohol Intake",
                    "active": "Physical Activity",
                    "bmi": "BMI",
                }
                display_field = field_map.get(field, field.replace("_", " ").title())
                return f"{display_field}: {value}"
        return feature_name

    def _local_shap_drivers(self, model: object, transformed_row, feature_names: List[str]) -> List[Dict[str, object]]:
        if shap is None or np is None:
            return []

        try:
            base_estimator = self._unwrap_base_estimator(model)
            class_name = base_estimator.__class__.__name__.lower()
            tree_like = any(
                token in class_name
                for token in ["xgb", "lgbm", "randomforest", "gradientboost", "decisiontree", "extratrees"]
            )
            if not tree_like:
                return []

            dense_row = transformed_row
            if hasattr(dense_row, "toarray"):
                dense_row = dense_row.toarray()
            dense_row = np.asarray(dense_row)

            explainer = shap.TreeExplainer(base_estimator)
            shap_values = explainer.shap_values(dense_row)

            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            row_values = np.asarray(shap_values)[0]
            top_indices = np.argsort(np.abs(row_values))[::-1][:3]

            return [
                {
                    "feature": feature_names[int(idx)],
                    "feature_label": self._friendly_feature_label(feature_names[int(idx)]),
                    "shap_contribution": float(row_values[int(idx)]),
                    "source": "local_prediction_shap",
                }
                for idx in top_indices
            ]
        except Exception:
            return []

    def _global_shap_fallback(self) -> List[Dict[str, object]]:
        if not os.path.exists(self.config.shap_summary_path):
            return []

        with open(self.config.shap_summary_path, "r", encoding="utf-8") as shap_file:
            shap_summary = json.load(shap_file)

        global_top = shap_summary.get("top_features_by_mean_abs_shap", [])
        fallback = []
        for item in global_top[:3]:
            fallback.append(
                {
                    "feature": item.get("feature"),
                    "feature_label": self._friendly_feature_label(str(item.get("feature"))),
                    "mean_abs_shap": item.get("mean_abs_shap"),
                    "source": "global_shap_summary",
                }
            )
        return fallback

    def predict(self, input_df: pd.DataFrame) -> Dict[str, object]:
        try:
            model = load_object(self.config.model_path)
            preprocessor = load_object(self.config.preprocessor_path)
            transformer = DataTransformation()
            decision_threshold = 0.5
            confidence_intervals = {}

            if os.path.exists(self.config.metrics_path):
                with open(self.config.metrics_path, "r", encoding="utf-8") as metrics_file:
                    metrics = json.load(metrics_file)
                    decision_threshold = float(metrics.get("decision_threshold", 0.5))
                    confidence_intervals = metrics.get("confidence_intervals_95", {})

            engineered_input_df = transformer._clean_and_engineer(input_df)
            transformed = preprocessor.transform(engineered_input_df)
            probability = float(model.predict_proba(transformed)[:, 1][0])
            risk_category = self._risk_category(probability)
            feature_names = self._load_feature_names(transformed.shape[1])
            top_drivers = self._local_shap_drivers(model, transformed[0:1], feature_names)
            if not top_drivers:
                top_drivers = self._global_shap_fallback()

            return {
                "risk_probability": probability,
                "risk_category": risk_category,
                "decision_threshold": decision_threshold,
                "predicted_class": int(probability >= decision_threshold),
                "confidence_intervals_95": confidence_intervals,
                "top_shap_drivers": top_drivers,
                "recommendations": self._recommendations(risk_category),
            }
        except Exception as e:
            raise CustomException(e, sys)
