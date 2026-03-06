import json
import os
import sys
from dataclasses import dataclass
from typing import Dict

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.utils import load_object


@dataclass
class PredictPipelineConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    metrics_path: str = os.path.join("artifacts", "metrics.json")

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

    def predict(self, input_df: pd.DataFrame) -> Dict[str, float]:
        try:
            model = load_object(self.config.model_path)
            preprocessor = load_object(self.config.preprocessor_path)
            transformer = DataTransformation()
            decision_threshold = 0.5

            if os.path.exists(self.config.metrics_path):
                with open(self.config.metrics_path, "r", encoding="utf-8") as metrics_file:
                    metrics = json.load(metrics_file)
                    decision_threshold = float(metrics.get("decision_threshold", 0.5))

            engineered_input_df = transformer._clean_and_engineer(input_df)
            transformed = preprocessor.transform(engineered_input_df)
            probability = float(model.predict_proba(transformed)[:, 1][0])
            return {
                "risk_probability": probability,
                "risk_category": self._risk_category(probability),
                "decision_threshold": decision_threshold,
                "predicted_class": int(probability >= decision_threshold),
            }
        except Exception as e:
            raise CustomException(e, sys)
