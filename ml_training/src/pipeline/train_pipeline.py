import json
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def main() -> None:
    try:
        logging.info("Training pipeline started.")

        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion complete. Train: %s Test: %s", train_data_path, test_data_path)

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data transformation complete. Preprocessor: %s", preprocessor_path)

        model_trainer = ModelTrainer()
        report = model_trainer.initiate_model_trainer(train_arr, test_arr)

        print("Training completed successfully.")
        print(json.dumps(report, indent=2))
        logging.info("Training pipeline completed.")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":

    cwd = os.getcwd()
    if os.path.basename(cwd) != "ml_training" and os.path.isdir(os.path.join(cwd, "ml_training")):
        os.chdir(os.path.join(cwd, "ml_training"))
    main()
