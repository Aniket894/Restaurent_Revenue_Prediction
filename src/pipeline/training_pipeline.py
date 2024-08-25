from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

def training_pipeline(input_file):
    # Data Ingestion
    data_ingestion = DataIngestion(input_file)
    train_file, test_file = data_ingestion.ingest_data()
    
    # Data Transformation
    data_transformation = DataTransformation(train_file, test_file)
    train_features, train_target, test_features, test_target = data_transformation.transform_data()
    
    # Model Training
    model_training = ModelTraining(train_features, train_target, test_features, test_target)
    model_training.train_and_evaluate_models()

if __name__ == "__main__":
    training_pipeline('notebook/data/weather_classification_data.csv')
