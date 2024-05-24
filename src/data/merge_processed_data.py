import json
import csv
import os
import pandas as pd
import holidays
from datetime import datetime


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')
raw_data_path = os.path.join(data_dir, 'raw')
processed_data_path = os.path.join(data_dir, 'processed')
merged_data_path = os.path.join(data_dir, 'merged')

slovenian_holidays = holidays.Slovenia()


def read_json(path, filename):
    filepath = os.path.join(path, filename)
    with open(filepath, mode='r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def slovenian_holiday(date):
    if date in slovenian_holidays:
        return slovenian_holidays[date]
    else:
        return "None"

def get_season_day_hour(date_str):
    # Convert string to datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

    # Get the season
    month = date.month
    if 3 <= month <= 5:
        season = "Spring"
    elif 6 <= month <= 8:
        season = "Summer"
    elif 9 <= month <= 11:
        season = "Autumn"
    else:
        season = "Winter"

    # Get the day of the week
    day_of_week = date.strftime("%A")

    # Get the hour
    hour = date.strftime("%H:%M:%S")

    return season, day_of_week, hour

def format_datetime(datetime_str):
    dt_obj = datetime.fromisoformat(datetime_str)   # Convert string to datetime object
    formatted_datetime = dt_obj.strftime('%d-%m-%Y')
    return formatted_datetime


def save_merged_data(row, filename):
    fieldnames = [
                    'date',   
                    'region',
                    'latitude',
                    'longitude',
                    'day_of_week',
                    'hour',
                    'season',
                    'fuel_price',
                    'holiday',
                    'temperature',
                    'relative_humidity',
                    'dew_point',
                    'surface_pressure',
                    'rain',
                    'snowfall',
                    'visibility',
                    'num_of_cars',
                    'avg_speed'
                ]
    filepath = os.path.join(merged_data_path, filename)

    # Check if file exists and is non-empty
    file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # Write header only if file is new/empty
        writer.writerow({
                'date': row['date'],   
                'region': row['region'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'day_of_week': row['day_of_week'],
                'hour': row['hour'],
                'season': row['season'],
                'fuel_price': row['fuel_price'],
                'holiday': row['holiday'],
                'temperature': row['temperature'],
                'relative_humidity': row['relative_humidity'],
                'dew_point': row['dew_point'],
                'surface_pressure': row['surface_pressure'],
                'rain': row['rain'],
                'snowfall': row['snowfall'],
                'visibility': row['visibility'],
                'num_of_cars': row['num_of_cars'],
                'avg_speed': row['avg_speed']
        })

def merge_data(fuel_data):
    fuel_price = fuel_data['results'][0]['prices']['95']

    # Get a list of all files in the 'processed_data_path' directory
    processed_files = os.listdir(processed_data_path)

    # Filter the traffic and weather files
    traffic_files = [f for f in processed_files if f.endswith("_traffic_data.csv")]
    weather_files = [f for f in processed_files if f.endswith("_weather_data.csv")]

    for traffic_file in traffic_files:
        region = traffic_file.replace('_traffic_data.csv', '')
        weather_file = f"{region}_weather_data.csv"
        if weather_file in weather_files:
            traffic_df = pd.read_csv(os.path.join(processed_data_path, traffic_file))
            weather_df = pd.read_csv(os.path.join(processed_data_path, weather_file))
            
            traffic_instance = traffic_df.iloc[-1]
            weather_instance = weather_df.iloc[-1]
            holiday = slovenian_holiday(weather_instance['datetime'])
            season, day_of_week, hour = get_season_day_hour(weather_instance['datetime'])

            merged_row = {
                'date': format_datetime(weather_instance['datetime']),   
                'region': region,
                'latitude': traffic_instance['latitude'],
                'longitude': traffic_instance['longitude'],
                'day_of_week': day_of_week,
                'hour': hour,
                'season': season,
                'fuel_price': fuel_price,
                'holiday': holiday,
                'temperature': weather_instance['temperature'],
                'relative_humidity': weather_instance['relative_humidity'],
                'dew_point': weather_instance['dew_point'],
                'surface_pressure': weather_instance['surface_pressure'],
                'rain': weather_instance['rain'],
                'snowfall': weather_instance['snowfall'],
                'visibility': weather_instance['visibility'],
                'num_of_cars': traffic_instance['num_of_cars'],
                'avg_speed': traffic_instance['avg_speed']
            }

            merged_file_name = f"{region}_data.csv"
            save_merged_data(merged_row, merged_file_name) 

    print("Data merged to CSV files successfully!")

    # merged_data_filepath = os.path.join(merged_data_path, f"station{station_number}_data.csv") 
    # with open(station_data_filepath, mode='r', encoding='utf-8') as station_file, \
    #      open(weather_data_filepath, mode='r', encoding='utf-8') as weather_file, \
    #      open(merged_data_filepath, mode='a', newline='', encoding='utf-8') as merged_file:
        
    #     station_reader = csv.DictReader(station_file)
    #     weather_reader = csv.DictReader(weather_file)
        
    #     merged_fieldnames = ['number', 
    #                          'datetime',
    #                          'name',
    #                          'address',
    #                          'coordinates', 
    #                          'temperature', 
    #                          'relative_humidity', 
    #                          'dew_point', 
    #                          'apparent_temperature', 
    #                          'precipitation_probability', 
    #                          'rain', 
    #                          'surface_pressure',
    #                          'bike_stands', 
    #                          'available_bike_stands'
    #                          ]
    #     merged_writer = csv.DictWriter(merged_file, fieldnames=merged_fieldnames)
    #     #merged_writer.writeheader()     # Header row with fieldnames
             
    #     last_station_row = None
    #     for station_row in station_reader:
    #         last_station_row = station_row
        
    #     last_weather_row = None
    #     for weather_row in weather_reader:
    #         last_weather_row = weather_row

    #     merged_row = {
    #         'number': last_station_row['number'],
    #         'datetime': last_weather_row['datetime'],
    #         'name': last_station_row['name'],
    #         'address': last_station_row['address'],
    #         'coordinates': last_station_row['coordinates'],
    #         'temperature': last_weather_row['temperature'],
    #         'relative_humidity': last_weather_row['relative_humidity'],
    #         'dew_point': last_weather_row['dew_point'],
    #         'apparent_temperature': last_weather_row['apparent_temperature'],
    #         'precipitation_probability': last_weather_row['precipitation_probability'],
    #         'rain': last_weather_row['rain'],
    #         'surface_pressure': last_weather_row['surface_pressure'],
    #         'bike_stands': last_station_row['bike_stands'],
    #         'available_bike_stands': last_station_row['available_bike_stands']
    #     }
    #     merged_writer.writerow(merged_row)

def main():
    fuel_json_filename = "raw_fuel_data.json"
    fuel_data = read_json(raw_data_path, fuel_json_filename)
    merge_data(fuel_data)

if __name__ == "__main__":
    main()