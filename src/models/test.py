import onnxruntime as rt
from mlflow import MlflowClient
import joblib
import mlflow
import onnx
import os
import dagshub
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')

# Setup MLflow and Dagshub
load_dotenv()
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
dagshub.auth.add_app_token(token=mlflow_password)
dagshub.init(repo_owner=mlflow_username, repo_name='trafficPredictions', mlflow=True)
mlflow.set_tracking_uri(mlflow_uri)


# try:
#     session = rt.InferenceSession("cars_model.onnx")
#     print("Model loaded successfully!")
#     # Attempt to run the model here
# except Exception as e:
#     print(f"Failed to load or run the model: {e}")


# Function for downloading models and scalers from MLflow
def download_models():
    mlflowClient = MlflowClient()
    print("Presaving models and scalers ...")

    for region in ['Kozina']:
        for model_type in ['cars']:
            model_name = f"{region}_{model_type}_model"
            target_scaler_name = f"{region}_{model_type}_target_scaler"
            features_scaler_name = f"{region}_{model_type}_features_scaler"

            model = mlflow.onnx.load_model(mlflowClient.get_latest_versions(name=model_name, stages=["Production"])[0].source)
            target_scaler = mlflow.sklearn.load_model(mlflowClient.get_latest_versions(name=target_scaler_name, stages=["Production"])[0].source)
            features_scaler = mlflow.sklearn.load_model(mlflowClient.get_latest_versions(name=features_scaler_name, stages=["Production"])[0].source)

            onnx.save_model(model, os.path.join(models_dir, region, model_type, f"{region}_{model_type}_model.onnx"))
            joblib.dump(target_scaler, os.path.join(models_dir, region, model_type, f'{region}_{model_type}_target_scaler.pkl'))
            joblib.dump(features_scaler, os.path.join(models_dir, region, model_type, f'{region}_{model_type}_features_scaler.pkl'))

    
    print("All models downloaded and saved locally!")

download_models()