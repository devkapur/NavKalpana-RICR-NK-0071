import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _clean_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        if "id" in data.columns:
            data = data.drop(columns=["id"])

        data["age_years"] = (data["age"] / 365.25).round(2)
        data["bmi"] = data["weight"] / ((data["height"] / 100.0) ** 2)
        data["pulse_pressure"] = data["ap_hi"] - data["ap_lo"]
        data["age_bp_interaction"] = data["age_years"] * data["ap_hi"]
        data["glucose_bmi_interaction"] = data["gluc"] * data["bmi"]

        # Robust clipping for noisy measurements often seen in field settings.
        clip_ranges = {
            "height": (120, 230),
            "weight": (30, 250),
            "ap_hi": (80, 240),
            "ap_lo": (40, 160),
            "bmi": (10, 60),
            "pulse_pressure": (10, 120),
            "age_years": (18, 90),
        }
        for col, (low, high) in clip_ranges.items():
            if col in data.columns:
                data[col] = data[col].clip(lower=low, upper=high)

        # Enforce DBP <= SBP after clipping.
        invalid_bp = data["ap_lo"] > data["ap_hi"]
        data.loc[invalid_bp, "ap_lo"] = data.loc[invalid_bp, "ap_hi"] - 10
        data["ap_lo"] = data["ap_lo"].clip(lower=40)

        data["bmi_risk_cat"] = pd.cut(
            data["bmi"],
            bins=[0, 18.5, 25, 30, 100],
            labels=["underweight", "normal", "overweight", "obese"],
        ).astype(str)

        return data

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numeric_features = [
                "age",
                "age_years",
                "height",
                "weight",
                "ap_hi",
                "ap_lo",
                "bmi",
                "pulse_pressure",
                "age_bp_interaction",
                "glucose_bmi_interaction",
            ]

            categorical_features = [
                "gender",
                "cholesterol",
                "gluc",
                "smoke",
                "alco",
                "active",
                "bmi_risk_cat",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            logging.info(f"Numerical columns: {numeric_features}  scaling and imputing complete.")
            logging.info(f"Categorical columns: {categorical_features} encoding complete.")


            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numeric_features),
                    ("cat", cat_pipeline, categorical_features),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = self._clean_and_engineer(train_df)
            test_df = self._clean_and_engineer(test_df)

            target_column_name = "cardio"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info("Data transformation completed and preprocessor saved.")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
