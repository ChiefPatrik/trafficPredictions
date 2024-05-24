import json
import csv
import os
import pandas as pd
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')
raw_data_path = os.path.join(data_dir, 'raw')
processed_data_path = os.path.join(data_dir, 'processed')


def read_json(path, filename):
    filepath = os.path.join(path, filename)
    with open(filepath, mode='r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def format_datetime(datetime_str):
    dt_obj = datetime.fromisoformat(datetime_str)   # Convert string to datetime object
    formatted_datetime = dt_obj.strftime('%Y-%m-%d %H:%M:%S%z')
    return formatted_datetime


# Function to find the matching weather data for a given region
def find_weather_data_for_region(weather_data, region):
    for entry in weather_data:
        if entry['region'] == region:
            return entry
    return None

# Function to find the closest index in the weather time array
def find_closest_weather_time(matching_weather, traffic_datetime):
    traffic_datetime = datetime.strptime(traffic_datetime, "%Y-%m-%dT%H:%M:%SZ")
    weather_times = matching_weather["hourly"]["time"]
    
    # Convert weather times to datetime objects
    weather_datetimes = [datetime.strptime(time, "%Y-%m-%dT%H:%M") for time in weather_times]
    
    # Calculate the absolute differences
    time_differences = [abs((traffic_datetime - weather_time).total_seconds()) for weather_time in weather_datetimes]
    
    # Find the index of the minimum difference
    closest_index = time_differences.index(min(time_differences))
    
    return closest_index


def save_weather_data(weather, filename):
    fieldnames = ['datetime',
                  'temperature', 
                  'relative_humidity', 
                  'dew_point', 
                  'surface_pressure',
                  'rain', 
                  'snowfall',
                  'visibility'
                  ]
    filepath = os.path.join(processed_data_path, filename)

    # Check if file exists and is non-empty
    file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0

    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # Write header only if file is new/empty
        writer.writerow({
            'datetime': format_datetime(weather['datetime']),
            'temperature': weather['temperature_2m'],
            'relative_humidity': weather['relative_humidity_2m'],
            'dew_point': weather['dew_point_2m'],
            'surface_pressure': weather['surface_pressure'],
            'rain': weather['rain'],
            'snowfall': weather['snowfall'],
            'visibility': weather['visibility']
        })

def process_data(weather_data):
    processed_files = os.listdir(processed_data_path)

    for traffic_file in processed_files:
        if traffic_file.endswith("_traffic_data.csv"):
            file_path = os.path.join(processed_data_path, traffic_file)
            traffic_df = pd.read_csv(file_path)
            traffic_datetime = traffic_df['datetime'].iloc[-1]
            traffic_region = traffic_df['region'].iloc[-1]

            # Find the matching weather for traffic data based on region
            matching_weather = find_weather_data_for_region(weather_data, traffic_region)

            # Find the index of the closest weather time to the traffic data time
            closest_index = find_closest_weather_time(matching_weather, traffic_datetime)
            weather_attributes = {
                'datetime': matching_weather['hourly']['time'][closest_index],
                'temperature_2m': matching_weather['hourly']['temperature_2m'][closest_index],
                'relative_humidity_2m': matching_weather['hourly']['relative_humidity_2m'][closest_index],
                'dew_point_2m': matching_weather['hourly']['dew_point_2m'][closest_index],
                'surface_pressure': matching_weather['hourly']['surface_pressure'][closest_index],
                'rain': matching_weather['hourly']['rain'][closest_index],
                'snowfall': matching_weather['hourly']['snowfall'][closest_index],
                'visibility': matching_weather['hourly']['visibility'][closest_index]
            }

            weather_filename = f"{traffic_region}_weather_data.csv"
            save_weather_data(weather_attributes, weather_filename)

    print("Weather data preprocessed and saved to CSV files successfully!")


def main():
    weather_json_filename = "raw_weather_data.json"
    weather_data = read_json(raw_data_path, weather_json_filename)
    process_data(weather_data)

if __name__ == "__main__":
    main()