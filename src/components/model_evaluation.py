import os, sys
import pandas as pd
from sklearn.metrics import f1_score

from src.logger import logging
from src.exception import CustomException
from src.constants import MODEL_DIR, MODEL_FILE_NAME, TFIDF_VECTORIZER_FILE_NAME
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.text_preprocessing import preprocess_text

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_test_data(self):
        test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
        test_df.dropna(inplace=True)
        test_df.drop_duplicates(inplace=True)

        texts = test_df["text"].apply(preprocess_text)
        encoder = load_object(
            self.data_transformation_artifact.target_encoder_file_path
        )
        y_true = encoder.transform(test_df["text_type"])
        return texts, y_true

    @staticmethod
    def _score(model, vectorizer, texts, y_true) -> float:
        return f1_score(
            y_true, model.predict(vectorizer.transform(texts)), average="weighted"
        )

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Model Evaluation started")

            texts, y_true = self._load_test_data()

            new_score = self._score(
                load_object(self.model_trainer_artifact.model_file_path),
                load_object(
                    self.data_transformation_artifact.tfidf_vectorizer_file_path
                ),
                texts,
                y_true,
            )

            prod_model = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
            prod_vec = os.path.join(MODEL_DIR, TFIDF_VECTORIZER_FILE_NAME)

            if os.path.exists(prod_model) and os.path.exists(prod_vec):
                prod_score = self._score(
                    load_object(prod_model), load_object(prod_vec), texts, y_true
                )
            else:
                logging.info("No production model found, accepting new model")
                prod_score = None

            difference = new_score - (prod_score or 0.0)
            artifact = ModelEvaluationArtifact(
                is_model_accepted=difference
                > self.model_eval_config.changed_threshold_score,
                new_model_score=new_score,
                production_model_score=prod_score,
                difference=difference,
            )
            logging.info(f"Model evaluation artifact: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys) from e
