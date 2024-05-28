import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split


current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
data_dir = os.path.join(current_dir, '..', '..', 'data')
merged_data_dir = os.path.join(data_dir, 'merged')


def encode_categorical_columns(df, columns):
    return pd.get_dummies(df, columns=columns)

def extract_hour(df, time_column):
    df['hour'] = df[time_column].apply(lambda x: int(x.split(':')[0]))
    return df

def train_and_save_model(region_name, df):
    categorical_columns = ['day_of_week', 'season', 'holiday']
    df = encode_categorical_columns(df, categorical_columns)
    df = extract_hour(df, 'hour')

    # Define the target and features
    target = 'num_of_cars'  # Assuming 'num_of_cars' is the target variable
    features = df.drop(columns=[target, 'date', 'region', 'latitude', 'longitude', 'avg_speed'])  # Drop non-feature columns

    X = features.values
    y = df[target].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=1234)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ev = explained_variance_score(y_test, y_pred)
    print(f'Region: {region_name}\n MSE: {mse} \n MAE: {mae} \n EV: {ev} \n\n')

    # Save the model to a file
    model_filename = os.path.join(models_dir, f'{region_name}_cars_model.pkl')
    joblib.dump(model, model_filename)
    print(f'Model saved to {model_filename}')

def main():
    
    for filename in os.listdir(merged_data_dir):
        if filename.endswith('_train.csv'):
            region_name = filename[:-10]    # Remove '_train.csv' (10 characters)
            file_path = os.path.join(merged_data_dir, filename)
            print(f'Processing file {region_name}_train.csv...')

            # Load the data
            traffic_df = pd.read_csv(file_path)

            # Train and save the model for the region
            train_and_save_model(region_name, traffic_df)

if __name__ == '__main__':
    main()