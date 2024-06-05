import os
import json
import holidays
from datetime import datetime
import onnx
import joblib
import dagshub
import uvicorn
import requests
import numpy as np
import pandas as pd
import onnxruntime as onnx_rt
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import mlflow
import mlflow.keras
import mlflow.sklearn
from mlflow.tracking import MlflowClient

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')
raw_data_path = os.path.join(data_dir, 'raw')
merged_data_dir = os.path.join(data_dir, 'merged')
models_dir = os.path.join(current_dir, '..', '..', 'models')


# Setup Dagshub, MLflow and MongoDB
load_dotenv()
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
mongo_username = os.getenv("MONGO_USERNAME")
mongo_password = os.getenv("MONGO_PASSWORD")

dagshub.auth.add_app_token(token=mlflow_password)
dagshub.init(repo_owner=mlflow_username, repo_name='trafficPredictions', mlflow=True)
mlflow.set_tracking_uri(mlflow_uri)

db_name = "trafficInputData"
connection_string = f"mongodb+srv://{mongo_username}:{mongo_password}@cluster0.ygb5z8f.mongodb.net/"
client = MongoClient(connection_string)


# Region names to numbers mapping
region_mapping = {
    'Kozina': 1,
    'Ljubljana': 2,
    'Maribor': 3,
    'Pomurska': 4,
    'Postojna': 5,
    'Sl_Konjice': 6,
    'Vransko': 7
}

# ===========================================
# DOWNLOADING, LOADING AND HELPER FUNCTIONS
# ===========================================

# Function for downloading models and scalers from MLflow
def download_models():
    mlflowClient = MlflowClient()
    print("Presaving models and scalers ...")

    for region in region_mapping.keys():
        for model_type in ['cars', 'speed']:
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

# Function for loading model and scalers from local files
def load_model_and_scalers(region, model_type):
    model_path = os.path.join(models_dir, region, model_type, f"{region}_{model_type}_model.onnx")
    target_scaler_path = os.path.join(models_dir, region, model_type, f"{region}_{model_type}_target_scaler.pkl")
    features_scaler_path = os.path.join(models_dir, region, model_type, f"{region}_{model_type}_features_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(target_scaler_path):
        raise FileNotFoundError(f"Target scaler file not found: {target_scaler_path}")
    if not os.path.exists(features_scaler_path):
        raise FileNotFoundError(f"Features scaler file not found: {features_scaler_path}")

    model = onnx_rt.InferenceSession(model_path)
    target_scaler = joblib.load(target_scaler_path)
    features_scaler = joblib.load(features_scaler_path)

    return model, target_scaler, features_scaler

# Function for getting saved predictions from MongoDB    
def get_predictions_from_mongo(): 
    db = client.get_database(db_name)  
    collection = db["trafficPredictions"]
    documents = collection.find()
    return list(documents)

# Function to find the index of a row in the dataset that matches the prediction date and hour
def find_index(traffic_data, prediction):
    # Convert 'date' column to datetime
    traffic_data['date'] = pd.to_datetime(traffic_data['date'], format='%d-%m-%Y')

    # Clean and convert 'hour' column to time
    traffic_data['hour'] = traffic_data['hour'].str.strip()
    traffic_data['hour'] = pd.to_datetime(traffic_data['hour'], format='%H:%M', errors='coerce').dt.time

    pred_date = datetime.strptime(prediction['date'], '%d-%m-%Y').date()
    pred_hour = datetime.strptime(prediction['hour'], '%H:%M').time()

    print(f"Prediction Date: {pred_date}")
    print(f"Prediction Hour: {pred_hour}")
    print("\nTraffic data:\n", traffic_data["date"])

    # Find the exact match for date and hour
    exact_match = traffic_data[(traffic_data['date'] == pred_date) & (traffic_data['hour'] == pred_hour)]
    print("\nExact Match:\n", exact_match)

    if not exact_match.empty:
        return exact_match.index[0]

    # Find the rows with the matching date
    date_match = traffic_data[traffic_data['date'] == pred_date]
    print("\nDate Match:\n", date_match)

    if date_match.empty:
        return "No matching date found."

    # Find the closest hour
    closest_hour = date_match.iloc[(date_match['hour'].apply(lambda x: abs(datetime.combine(pred_date, x) - datetime.combine(pred_date, pred_hour))).argsort())].iloc[0]
    print("\nClosest Hour Match:\n", closest_hour)

    return closest_hour.name


# ===========================================
# EVALUATION FUNCTIONS
# ===========================================

# Function for evaluating "num_of_cars" predictions with actual values and logging metrics to MLflow
def evaluate_cars_prediction(traffic_data, index, prediction):
    print(f"Evaluating 'num_of_cars' prediction for {prediction['region']} on {prediction['date']} ...")

    model, target_scaler, features_scaler = load_model_and_scalers(prediction['region'], 'cars')
    y_pred = prediction['num_of_cars']
    y_test = traffic_data['num_of_cars'].iloc[index]
    print(f"predicted: {y_pred}, actual: {y_test}")

    y_pred = np.array(y_pred).reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # mlflow.set_experiment(f"{traffic_data['region']}_cars_predictions_exp")
    # with mlflow.start_run(run_name=f"{traffic_data['region']}_cars_predictions_run"):
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    y_test_inv = target_scaler.inverse_transform(y_test)

    # Calculating metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    ev = explained_variance_score(y_test_inv, y_pred_inv)  
    print(f"MSE: {mse}, MAE: {mae}, EVS: {ev}\n")

    #     mlflow.log_metric("MAE", mae)
    #     mlflow.log_metric("MSE", mse)
    #     mlflow.log_metric("EVS", ev)
    # mlflow.end_run()

# Function for evaluating "avg_speed" predictions with actual values and logging metrics to MLflow
def evaluate_speed_prediction(traffic_data, index, prediction):
    print(f"Evaluating 'avg_speed' prediction for {prediction['region']} on {prediction['date']} ...")

    model, target_scaler, features_scaler = load_model_and_scalers(prediction['region'], 'speed')
    y_pred = prediction['avg_speed']
    y_test = traffic_data['avg_speed'].iloc[index]
    print(f"predicted: {y_pred}, actual: {y_test}")

    y_pred = np.array(y_pred).reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # mlflow.set_experiment(f"{traffic_data['region']}_speed_predictions_exp")
    # with mlflow.start_run(run_name=f"{traffic_data['region']}_speed_predictions_run"):
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    y_test_inv = target_scaler.inverse_transform(y_test)

    # Calculating metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    ev = explained_variance_score(y_test_inv, y_pred_inv)
    print(f"MSE: {mse}, MAE: {mae}, EVS: {ev}\n")

    #     mlflow.log_metric("MAE", mae)
    #     mlflow.log_metric("MSE", mse)
    #     mlflow.log_metric("EVS", ev)
    # mlflow.end_run()

def main():
    #download_models()
    #predictions = get_predictions_from_mongo()
    predictions = [
        {
            "region": "Ljubljana",
            "date": "02-06-2024",
            "hour": "15:00",
            "num_of_cars": 180,
            "avg_speed": 88
        },
        {
            "region": "Vransko",
            "date": "03-06-2024",
            "hour": "14:00",
            "num_of_cars": 190,
            "avg_speed": 92
        },
        {
            "region": "Kozina",
            "date": "02-06-2024",
            "hour": "13:00",
            "num_of_cars": 210,
            "avg_speed": 89
        },
        {
            "region": "Maribor",
            "date": "01-06-2024",
            "hour": "12:00",
            "num_of_cars": 195,
            "avg_speed": 91
        },
        {
            "region": "Vransko",
            "date": "29-05-2024",
            "hour": "11:00",
            "num_of_cars": 202,
            "avg_speed": 90
        }
    ]

    for prediction in predictions:
        region = prediction['region']

        # Get the corresponding CSV file
        filename = f"{region}_data.csv"
        filepath = os.path.join(merged_data_dir, filename)
        traffic_data = pd.read_csv(filepath)

        # Find the index of the row that matches the "date" and "hour" from the prediction
        index = find_index(traffic_data, prediction)
        #print("Index: ", index)
        if index == "No matching date found.":
            print(f"Did not find the actual value of prediction for {region} on {prediction['date']} in dataset.")
            continue
        #evaluate_cars_prediction(traffic_data, index, prediction)
        #evaluate_speed_prediction(traffic_data, index, prediction)


if __name__ == "__main__":
    main()