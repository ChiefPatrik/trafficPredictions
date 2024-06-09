import os
import joblib
import onnx
import tf2onnx
import mlflow
import dagshub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature


current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
data_dir = os.path.join(current_dir, '..', '..', 'data')
merged_data_dir = os.path.join(data_dir, 'merged')

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
    # df.to_csv("processed_dataset.csv", index=False)
    return df


def build_model(input_dim):
    # model = Sequential()
    # model.add(Dense(128, input_dim=input_dim, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    
    # model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    
    # model.add(Dense(32, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    
    # model.add(Dense(1, activation='linear'))
    
    # model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Define the pruning schedule
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.2,
            final_sparsity=0.8,
            begin_step=0,
            end_step=1000
        )
    }

    inputs = tf.keras.Input(shape=(input_dim,))
    #x = Dense(128, activation='relu')(inputs)
    x = prune_low_magnitude(Dense(128, activation='relu'), **pruning_params)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    #x = Dense(64, activation='relu')(x)
    x = prune_low_magnitude(Dense(64, activation='relu'), **pruning_params)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    #x = Dense(32, activation='relu')(x)
    x = prune_low_magnitude(Dense(32, activation='relu'), **pruning_params)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae', 'mse']) 
    
    return model

# Function for saving the model and scalers to file
def save_model_scalers(region_name, model, features_scaler, target_scaler):
    save_dir = os.path.join(models_dir, region_name, "cars")
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    onnx.save_model(model, os.path.join(save_dir, f'{region_name}_cars_model.onnx'))
    joblib.dump(target_scaler, os.path.join(save_dir, f'{region_name}_cars_target_scaler.pkl'))
    joblib.dump(features_scaler, os.path.join(save_dir, f'{region_name}_cars_features_scaler.pkl'))
    print(f'<Cars> Model and scalers for {region_name} saved successfully!')

# Function for saving scalers to MLflow
def mlflow_save_scaler(client, scaler_name, scaler, region_name, stage_param="Staging"):
    metadata = {
        "region_number": region_name,
        "scaler_name": scaler_name,
        "expected_features": scaler.n_features_in_,
        "feature_range": scaler.feature_range,
    }

    scaler = mlflow.sklearn.log_model(
        sk_model=scaler,
        artifact_path=f"models/{region_name}/{scaler_name}",
        registered_model_name=f"{region_name}_{scaler_name}",
        metadata=metadata,
    )

    scaler_version = client.create_model_version(
        name=f"{region_name}_{scaler_name}",
        source=scaler.model_uri,
        run_id=scaler.run_id
    )

    client.transition_model_version_stage(
        name=f"{region_name}_{scaler_name}",
        version=scaler_version.version,
        stage=stage_param,
    )


def get_metric_from_run(client, run_id, metric_name):
    metric = client.get_metric_history(run_id, metric_name)
    return metric[-1].value if metric else None

def train_model(region_name, traffic_df, client):
    traffic_df = process_data(traffic_df)
    
    # Define the target and features
    target = 'num_of_cars' 
    features = traffic_df.drop(columns=[target, 'latitude', 'longitude', 'avg_speed'])  

    # Normalize data
    features_scaler = MinMaxScaler()
    features = features_scaler.fit_transform(features)

    target_scaler = MinMaxScaler()
    traffic_df[target] = target_scaler.fit_transform(traffic_df[[target]])

    X = features
    y = traffic_df[target].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    experiment_name = f"{region_name}_train_cars_exp"
    model_name = f"{region_name}_cars_model"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{region_name}_train_cars_run"):
        mlflow.autolog()
        run_id = mlflow.active_run().info.run_id    # For versioning mlflow models

        # Define the pruning callbacks
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir='./logs')
        ]

        # Build and train model
        model = build_model(X_train.shape[1])
        train_history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2, verbose=1, callbacks=callbacks)
        predictions = model.predict(X_test)

        # Log train metrics to MLflow
        for i in range(len(train_history.history['loss'])):
            mlflow.log_metric("loss", train_history.history['loss'][i], step=i)
            mlflow.log_metric("val_loss", train_history.history['val_loss'][i], step=i)

        # Convert the model to ONNX format
        input_signature = [
            tf.TensorSpec(shape=(None, X_train.shape[1]), dtype=tf.float32, name="input")
        ]
        onnx_model, _ = tf2onnx.convert.from_keras(model=model, input_signature=input_signature, opset=13)  
        
        save_model_scalers(region_name, onnx_model, features_scaler, target_scaler)

        # Log the model to MLflow
        registered_model = mlflow.onnx.log_model(onnx_model=onnx_model, 
                                artifact_path=f"models/{region_name}/model", 
                                signature=infer_signature(X_test, predictions), 
                                registered_model_name=model_name)
        model_version = client.create_model_version(name=model_name, source=registered_model.model_uri, run_id=run_id)    

        try:
            model_versions = client.get_latest_versions(model_name, stages=["Production"])
            if model_versions:
                last_production_version = model_versions[0]
                last_production_run_id = last_production_version.run_id

                # Retrieve metrics of the last production model
                production_mse = get_metric_from_run(client, last_production_run_id, "mse")

                # Retrieve metrics of the current model
                current_mse = get_metric_from_run(client, run_id, "mse")

                print(f"Current model mse: {current_mse}")
                print(f"Production model mse: {production_mse}")

                if current_mse < production_mse:
                    print("Current model is better. Transitioning it to production.")
                    client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Production"
                    )
                    mlflow_save_scaler(client, "cars_target_scaler", target_scaler, region_name, "Production")
                    mlflow_save_scaler(client, "cars_features_scaler", features_scaler, region_name, "Production")
                else:
                    print("Current model is not better than the production model. Transitioning to staging.")
                    client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Staging"
                    )
                    mlflow_save_scaler(client, "cars_target_scaler", target_scaler, region_name, "Staging")
                    mlflow_save_scaler(client, "cars_features_scaler", features_scaler, region_name, "Staging")
            else:
                print("No production model found. Transitioning current model to production.")
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Production"
                )
                mlflow_save_scaler(client, "cars_target_scaler", target_scaler, region_name, "Production")
                mlflow_save_scaler(client, "cars_features_scaler", features_scaler, region_name, "Production")
        except Exception as e:
            print(f"Error while comparing models: {e}")
            print("Transitioning current model to staging.")
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            mlflow_save_scaler(client, "cars_target_scaler", target_scaler, region_name, "Staging")
            mlflow_save_scaler(client, "cars_features_scaler", features_scaler, region_name, "Staging")

    mlflow.end_run()



def main():
    client = MlflowClient()
    for filename in os.listdir(merged_data_dir):
        if filename.endswith('_train.csv'):
            file_path = os.path.join(merged_data_dir, filename)
            print(f'Processing file {filename}...')

            # Load the data
            traffic_df = pd.read_csv(file_path)
            region_name = filename[:-10]    # Remove '_train.csv' (10 characters)

            # Build and train the model for the region
            train_model(region_name, traffic_df, client)
            break

    # for region in region_mapping.keys():
    #     client.delete_registered_model(name=f"{region}_cars_model")
    #     client.delete_registered_model(name=f"{region}_cars_features_scaler")
    #     client.delete_registered_model(name=f"{region}_cars_target_scaler")
    #     client.delete_registered_model(name=f"{region}_speed_model")
    #     client.delete_registered_model(name=f"{region}_speed_features_scaler")
    #     client.delete_registered_model(name=f"{region}_speed_target_scaler")



if __name__ == '__main__':
    main()