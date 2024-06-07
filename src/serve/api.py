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


# ======================================
# DATA PREPROCESSING FUNCTIONS
# ======================================

def read_json(path, filename):
    filepath = os.path.join(path, filename)
    with open(filepath, mode='r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def slovenian_holiday(date):
    slovenian_holidays = holidays.Slovenia()
    if date in slovenian_holidays:
        return slovenian_holidays[date]
    else:
        return "None"

def get_season_day_hour(date_str):
    # Convert string to datetime object
    date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M')

    # Get the season
    month = date.month
    if 3 <= month <= 5:
        season = "Spring"
    elif 6 <= month <= 8:
        season = "Summer"
    elif 9 <= month <= 11:
        season = "Fall"
    else:
        season = "Winter"

    # Get the day of the week
    day_of_week = date.strftime("%A")

    # Get the hour
    hour = date.strftime("%H:%M")

    return season, day_of_week, hour

def format_datetime(datetime_str):
    dt_obj = datetime.fromisoformat(datetime_str)   # Convert string to datetime object
    formatted_datetime = dt_obj.strftime('%d-%m-%Y')
    return formatted_datetime


def fetch_fresh_weather_data(coordinates_lat, coordinates_lng):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coordinates_lat}&longitude={coordinates_lng}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,rain,snowfall,visibility&forecast_days=14&timezone=auto"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def extract_weather_info(weather_data, datetime_str):
    # Find the matching datetime in the weather data
    try:
        time_index = weather_data['hourly']['time'].index(datetime_str)
    except ValueError:
        raise ValueError(f"No weather data found for datetime: {datetime_str}")

    # Extract and aggregate weather attributes at the found index
    weather_info = {
        "datetime": datetime_str,
        "temperature": weather_data['hourly']['temperature_2m'][time_index],
        "relative_humidity": weather_data['hourly']['relative_humidity_2m'][time_index],
        "dew_point": weather_data['hourly']['dew_point_2m'][time_index],
        "surface_pressure": weather_data['hourly']['surface_pressure'][time_index],
        "rain": weather_data['hourly']['rain'][time_index],
        "snowfall": weather_data['hourly']['snowfall'][time_index],
        "visibility": weather_data['hourly']['visibility'][time_index]
    }
    return weather_info

def construct_prediction_input(region, date, hour):
    datetime_str = f"{date}T{hour}"

    # Retrieve the coordinates for the specified region
    coordinates = cities_coordinates.get(region)
    if coordinates is None:
        raise ValueError(f"Region '{region}' not found in cities_coordinates")
    
    # Fetch fresh weather data for the specified coordinates
    weather_data = fetch_fresh_weather_data(coordinates['latitude'], coordinates['longitude'])
    # Extract weather info for the specified datetime
    weather_info = extract_weather_info(weather_data, datetime_str)

    # Get fuel price
    fuel_price = read_json(raw_data_path, "raw_fuel_data.json")['results'][0]['prices']['95']

    # Get the season, day of the week and holiday for the specified datetime
    season, day_of_week, _ = get_season_day_hour(datetime_str)

    # Get the holiday for the specified datetime
    holiday = slovenian_holiday(datetime_str)

    # Format the datetime and hour same as in the training data
    hour = hour + ":00"
    formatted_datetime = format_datetime(datetime_str)

    prediction_input = {
        "date": formatted_datetime,
        "region": region,
        "day_of_week": day_of_week,
        "hour": hour,
        "season": season,
        "fuel_price": fuel_price,
        "holiday": holiday,
        "temperature": weather_info["temperature"],
        "relative_humidity": weather_info["relative_humidity"],
        "dew_point": weather_info["dew_point"],
        "surface_pressure": weather_info["surface_pressure"],
        "rain": weather_info["rain"],
        "snowfall": weather_info["snowfall"],
        "visibility": weather_info["visibility"]
    }
    # print(prediction_input)

    return prediction_input



# ======================================
# DATA PROCESSING FUNCTIONS
# ======================================

# Region names to coordinates mapping
cities_coordinates = {
    "Ljubljana": {"latitude": 46.07061503201507, "longitude": 14.577867970254866},
    "Maribor": {"latitude": 46.68255971126553, "longitude": 15.65138919777721},
    "Sl_Konjice": {"latitude": 46.25413437290015, "longitude": 15.302557315050453},
    "Postojna": {"latitude": 45.93134443045748, "longitude": 14.270708378492925},
    "Vransko": {"latitude": 46.174640576447764, "longitude": 14.804130481638964},
    "Pomurska": {"latitude": 46.52351975291412, "longitude": 16.44175950632071},
    "Kozina": {"latitude": 45.60742223894982, "longitude": 13.927767896289717}
}

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

# Categories values for selected columns
categories_dict = {
    'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'season': ['Spring', 'Summer', 'Fall', 'Winter'],
}

# Function to get dummies with specified categories
def get_dummies_with_all_categories(df, columns):
    for column in columns:
        df[column] = pd.Categorical(df[column], categories=categories_dict[column])
    return pd.get_dummies(df, columns=columns)

# Function to convert holiday values to binary
def preprocess_holiday_column(df, column_name='holiday'):
    df[column_name] = df[column_name].apply(lambda x: 1 if x == 'None' else 0)
    return df

# Function to process data same way it was processed during training
def process_data(df):
    # Encode categorical columns
    # df = pd.get_dummies(df, columns=['day_of_week', 'season', 'holiday'])    
    df = get_dummies_with_all_categories(df, ['day_of_week', 'season'])
    df = preprocess_holiday_column(df)

    # Extract hour from time column
    df['hour'] = df['hour'].apply(lambda x: int(x.split(':')[0]))       

    # Extract date features
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')      
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop(columns=['date'])

    # Encode region names to numbers
    region_name = df['region'].iloc[-1]
    region_number = region_mapping.get(region_name, 0)  # Default to 0 if region name not found
    df['region'] = region_number

    # Convert integer columns to float32
    int_columns = df.select_dtypes(include=['int64', 'int32']).columns
    df[int_columns] = df[int_columns].astype(np.float32)

    # Convert boolean columns to float32
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(np.float32)

    # print(df.dtypes)
    # print(df.head())
    return df



# ===========================================
# DOWNLOADING, LOADING AND SAVING FUNCTIONS
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

# Function for saving input data to MongoDB    
def save_to_mongodb(region, date, hour, num_of_cars, avg_speed): 
    db = client.get_database(db_name)  
    collection = db["trafficPredictions"]
    
    # Convert the DataFrame to a list of dictionaries for insertion into MongoDB
    data_dict = {
        "region": region,
        "date": date,
        "hour": hour,
        "num_of_cars": num_of_cars,
        "avg_speed": avg_speed
    }
    print("Input data to save to MongoDB:", data_dict)

    collection.insert_one(data_dict)
    print("Input data saved to MongoDB!")

# Function for getting 'num_of_cars' evaluation info from MLflow
def get_cars_evaluation_info(region):
    client = MlflowClient()

    # GET THE TRAIN RUN PARAMS
    run_name_pattern = f"{region}_train_cars_run"
    experiment_name = f"{region}_train_cars_exp"

    # Search for runs that match the run name pattern
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string=f"tags.mlflow.runName LIKE '%{run_name_pattern}%'",
        order_by=["start_time DESC"],
        max_results=25
    )

    # Get the latest 20 train run metrics
    metrics = []
    for run in runs:
        if run.data.metrics:
            metrics.append(run.data.metrics)
    
    # Get the latest train run parameters
    if runs:
        latest_run = runs[0]
        train_obj = {
            "run_id": latest_run.info.run_id,
            "start_time": latest_run.info.start_time,
            "status": latest_run.info.status,
            "metrics": latest_run.data.metrics,
            "params": latest_run.data.params,
            "tags": latest_run.data.tags,
            "artifact_uri": latest_run.info.artifact_uri
        }
        params = train_obj["params"]

        # GET THE LATEST EVALUATION RUN METRICS
        run_name_pattern = f"{region}_eval_cars_run"
        experiment_name = f"{region}_eval_cars_exp"

        # Search for runs that match the run name pattern
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        runs = client.search_runs(
            experiment_ids=experiment_id,
            filter_string=f"tags.mlflow.runName LIKE '%{run_name_pattern}%'",
            order_by=["start_time DESC"],
            max_results=25
        )

        # Get the latest evaluation run
        if runs:
            latest_run = runs[0]
            eval_obj = {
                "run_id": latest_run.info.run_id,
                "start_time": latest_run.info.start_time,
                "status": latest_run.info.status,
                "metrics": latest_run.data.metrics,
                "params": latest_run.data.params,
                "tags": latest_run.data.tags,
                "artifact_uri": latest_run.info.artifact_uri
            }
            latest_eval_metrics = eval_obj["metrics"]

            return params, metrics, latest_eval_metrics
    else:
        return None

# Function for getting 'avg_speed' evaluation info from MLflow
def get_speed_evaluation_info(region):
    client = MlflowClient()

    # GET THE TRAIN RUN PARAMS
    run_name_pattern = f"{region}_train_speed_run"
    experiment_name = f"{region}_train_speed_exp"

    # Search for runs that match the run name pattern
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string=f"tags.mlflow.runName LIKE '%{run_name_pattern}%'",
        order_by=["start_time DESC"],
        max_results=25
    )

    # Get the latest 20 train run metrics
    metrics = []
    for run in runs:
        if run.data.metrics:
            metrics.append(run.data.metrics)
    
    # Get the latest train run parameters
    if runs:
        latest_run = runs[0]
        train_obj = {
            "run_id": latest_run.info.run_id,
            "start_time": latest_run.info.start_time,
            "status": latest_run.info.status,
            "metrics": latest_run.data.metrics,
            "params": latest_run.data.params,
            "tags": latest_run.data.tags,
            "artifact_uri": latest_run.info.artifact_uri
        }
        params = train_obj["params"]

        # GET THE LATEST EVALUATION RUN METRICS
        run_name_pattern = f"{region}_eval_speed_run"
        experiment_name = f"{region}_eval_speed_exp"

        # Search for runs that match the run name pattern
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        runs = client.search_runs(
            experiment_ids=experiment_id,
            filter_string=f"tags.mlflow.runName LIKE '%{run_name_pattern}%'",
            order_by=["start_time DESC"],
            max_results=25
        )

        # Get the latest evaluation run
        if runs:
            latest_run = runs[0]
            eval_obj = {
                "run_id": latest_run.info.run_id,
                "start_time": latest_run.info.start_time,
                "status": latest_run.info.status,
                "metrics": latest_run.data.metrics,
                "params": latest_run.data.params,
                "tags": latest_run.data.tags,
                "artifact_uri": latest_run.info.artifact_uri
            }
            latest_eval_metrics = eval_obj["metrics"]

            return params, metrics, latest_eval_metrics
    else:
        return None
    

# ======================================
# DIRECT PREDICTION FUNCTIONS
# ======================================

def predict_cars(cars_input, region):
    # Load the model and scalers
    model, target_scaler, features_scaler = load_model_and_scalers(region, "cars")
    # Transform the input data
    features = features_scaler.transform(cars_input)

    # Run the model prediction
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    pred_onx = model.run([output_name], {input_name: features.astype(np.float32)})[0]

    # Inverse transform predictions
    y_pred = target_scaler.inverse_transform(pred_onx)
    return y_pred[0]

def predict_speed(cars_input, region, num_of_cars):
    # Load the model and scalers
    model, target_scaler, features_scaler = load_model_and_scalers(region, "speed")
    # Add the number of cars to the input data (on same index 11 as it was at training) and transform it
    cars_input.insert(11, 'num_of_cars', num_of_cars)
    features = features_scaler.transform(cars_input)

    # Run the model prediction
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    pred_onx = model.run([output_name], {input_name: features.astype(np.float32)})[0]

    # Inverse transform predictions
    y_pred = target_scaler.inverse_transform(pred_onx)
    return y_pred[0]

def predict(region, date, hour):
    print("Predicting traffic for", region, "on", date, "at", hour)
    # Construct and preprocess the prediction input
    cars_input = construct_prediction_input(region, date, hour)
    cars_input_df = pd.DataFrame([cars_input])
    formated_date = cars_input_df["date"][0]

    # Process the input data same way it was processed during training
    cars_input_df = process_data(cars_input_df)

    num_of_cars = predict_cars(cars_input_df, region)
    avg_speed = predict_speed(cars_input_df, region, num_of_cars)
    print("Predicted number of cars:", num_of_cars)
    print("Predicted average speed:", avg_speed)

    return round(num_of_cars[0]), round(avg_speed[0]), formated_date


# ======================================
# FASTAPI SERVER
# ======================================

# Create server
app = FastAPI()
# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrafficData(BaseModel):
    region: str
    date: str
    hour: str

class PredictionInput(BaseModel):
    data: List[TrafficData]


class ModelData(BaseModel):
    region: str

class MlflowInput(BaseModel):
    data: List[ModelData]


@app.post("/traffic/predict/", response_model=dict)
async def predict_traffic(input_data: PredictionInput):
    region = input_data.data[0].region
    date = input_data.data[0].date
    hour = input_data.data[0].hour
    try:   
        num_of_cars, avg_speed, formated_date = predict(region, date, hour)
        response = {
            "num_of_cars": num_of_cars,
            "avg_speed": avg_speed
        }
        save_to_mongodb(region, formated_date, hour, num_of_cars, avg_speed)
        return {'predictions': response} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/traffic/evaluation/", response_model=dict)
async def evaluate_models(input_data: MlflowInput):
    region = input_data.data[0].region
    try:   
        params_cars, metrics_cars, latest_eval_metrics_cars = get_cars_evaluation_info(region)
        params_speed, metrics_speed, latest_eval_metrics_speed = get_speed_evaluation_info(region)
        response = {
            "cars_evaluation": {
                "params": params_cars,
                "metrics": metrics_cars,
                "eval_metrics": latest_eval_metrics_cars
            },
            "speed_evaluation": {
                "params": params_speed,
                "metrics": metrics_speed,
                "eval_metrics": latest_eval_metrics_speed
            }
        }
        return {'evaluation': response} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    


if __name__ == "__main__":
    download_models()
    uvicorn.run(app, host="0.0.0.0", port=3001)