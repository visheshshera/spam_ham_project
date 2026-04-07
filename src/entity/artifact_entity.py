from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str



@dataclass
class DataTransformationArtifact:
    x_train_resampled_path: str
    y_train_resampled_path: str
    x_test_path: str
    y_test_path: str

@dataclass
class ModelTrainerArtifact:
    model_file_path: str