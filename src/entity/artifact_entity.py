from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str


@dataclass
class DataTransformationArtifact:
    x_train_resampled_path: str
    y_train_resampled_path: str
    x_test_path: str
    y_test_path: str
    tfidf_vectorizer_file_path: str
    target_encoder_file_path: str


@dataclass
class ModelTrainerArtifact:
    model_file_path: str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    new_model_score: float
    production_model_score: Optional[float]  # None if no production model yet
    difference: float


@dataclass
class ModelPusherArtifact:
    model_file_path: str
    vectorizer_file_path: str
    metadata_file_path: str
