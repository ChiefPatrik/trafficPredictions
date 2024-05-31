import os
import mlflow
import dagshub
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import onnx
import dagshub
from mlflow import MlflowClient

current_dir = os.path.dirname(os.path.abspath(__file__))
merged_data_dir = os.path.join(current_dir, '..', '..', 'data', 'merged')
models_dir = os.path.join(current_dir, '..', '..', 'models')
reports_dir = os.path.join(current_dir, '..', '..', 'reports')

# Setup MLflow and Dagshub
load_dotenv()
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
dagshub.auth.add_app_token(token=mlflow_password)
dagshub.init(repo_owner=mlflow_username, repo_name='trafficPredictions', mlflow=True)
mlflow.set_tracking_uri(mlflow_uri)

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

# Define all possible categories for categorical columns
day_of_week_categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
season_categories = ['Spring', 'Summer', 'Fall', 'Winter']

# Define the categories for each column
categories_dict = {
    'day_of_week': day_of_week_categories,
    'season': season_categories,
}

# Function to get dummies with specified categories
def get_dummies_with_all_categories(df, columns):
    for column in columns:
        df[column] = pd.Categorical(df[column], categories=categories_dict[column])
    return pd.get_dummies(df, columns=columns)

def preprocess_holiday_column(df, column_name='holiday'):
    df[column_name] = df[column_name].apply(lambda x: 1 if x == 'None' else 0)
    return df

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

def load_test_data(region_name):
    test_file = os.path.join(merged_data_dir, f"{region_name}_test.csv")
    if os.path.exists(test_file):
        return pd.read_csv(test_file)
    else:
        print(f"Test data file not found for {region_name}")
        return None
    
def save_data(model_type, region, mse, mae, ev):
    report_dir = os.path.join(reports_dir, region)
    report_file = os.path.join(report_dir, f"{model_type}_evaluation_report.csv")
    
    # Create the directory if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    if os.path.exists(report_file):
        report = pd.read_csv(report_file)
    else:
        report = pd.DataFrame(columns=['region', 'model', 'mse', 'mae', 'ev'])

    report = report._append({
        'region': region,
        'model': model_type,
        'mse': mse,
        'mae': mae,
        'ev': ev
    }, ignore_index=True)

    report.to_csv(report_file, index=False)
    
    print(f"Metrics for model '{model_type}' for {region} saved successfully!\n")



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

def load_model_and_scalers(model_dir, region, model_type):
    model_path = os.path.join(model_dir, f"{region}_{model_type}_model.onnx")
    target_scaler_path = os.path.join(model_dir, f"{region}_{model_type}_target_scaler.pkl")
    features_scaler_path = os.path.join(model_dir, f"{region}_{model_type}_features_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(target_scaler_path):
        raise FileNotFoundError(f"Target scaler file not found: {target_scaler_path}")
    if not os.path.exists(features_scaler_path):
        raise FileNotFoundError(f"Features scaler file not found: {features_scaler_path}")
    

    # onnx_model = rt.InferenceSession(os.path.join(models_dir, region_name, "cars", "cars_model.onnx"))
    # # Inspect the input and output names of the ONNX model
    # input_names = [input.name for input in onnx_model.get_inputs()]
    # output_names = [output.name for output in onnx_model.get_outputs()]
    # print("Input names:", input_names)
    # print("Output names:", output_names)

    # # Use the correct input name for predictions
    # input_name = input_names[0]  # Assuming the first input is the one we want
    # output_name = output_names[0]  # Assuming the first output is the one we want

    # # Run the model prediction
    # pred_onx = onnx_model.run([output_name], {input_name: X_test.astype(np.float32)})[0]
    # print("ONNX model prediction: ", pred_onx[:5])


    model = rt.InferenceSession(model_path)
    target_scaler = joblib.load(target_scaler_path)
    features_scaler = joblib.load(features_scaler_path)

    return model, target_scaler, features_scaler


def evaluate_model(model, target_scaler, features_scaler, test_data, model_type, region):
    test_data = process_data(test_data)
    
    if(model_type == 'cars'):
        target = 'num_of_cars' 
        features = test_data.drop(columns=[target, 'latitude', 'longitude', 'avg_speed'])  
    else:
        target = 'avg_speed'
        features = test_data.drop(columns=[target, 'latitude', 'longitude'])  

    # Normalize data
    features = features_scaler.transform(features)
    test_data[target] = target_scaler.transform(test_data[[target]])

    X_test = features
    y_true = test_data[target].values

    mlflow.set_experiment(f"{region}_eval_{model_type}_exp")
    with mlflow.start_run(run_name=f"{region}_eval_{model_type}_run"):
        # Run the model prediction
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        pred_onx = model.run([output_name], {input_name: X_test.astype(np.float32)})[0]

        # Inverse transform predictions
        y_pred = target_scaler.inverse_transform(pred_onx)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("evs", ev)

    mlflow.end_run()
    return mse, mae, ev


def evaluate_all_models():
    for region in os.listdir(models_dir):
        region_dir = os.path.join(models_dir, region)
        if os.path.isdir(region_dir):
            for model_type in ['cars', 'speed']:
                model_dir = os.path.join(region_dir, model_type)
                if os.path.exists(model_dir):
                    print(f"Evaluating model '{model_type}' for {region} ...")
                    test_data = load_test_data(region)
                    if test_data is not None:
                        model, target_scaler, features_scaler = load_model_and_scalers(model_dir, region, model_type)
                        mse, mae, ev = evaluate_model(model, target_scaler, features_scaler, test_data, model_type, region)
                        print(f"MSE: {mse}, MAE: {mae}, EV: {ev}")
                        save_data(model_type, region, mse, mae, ev)
                else:
                    print(f"Model directory not found: {model_dir}")

if __name__ == "__main__":
    #download_models()
    evaluate_all_models()