import requests
import json
import os
import pandas as pd
from math import radians, cos, sin, sqrt, atan2

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')
raw_data_path = os.path.join(data_dir, 'raw')
processed_data_path = os.path.join(data_dir, 'processed')

cities_coordinates = {
    "Ljubljana": {"latitude": 46.07061503201507, "longitude": 14.577867970254866},
    "Maribor": {"latitude": 46.68255971126553, "longitude": 15.65138919777721},
    "Sl_Konjice": {"latitude": 46.25413437290015, "longitude": 15.302557315050453},
    "Postojna": {"latitude": 45.93134443045748, "longitude": 14.270708378492925},
    "Vransko": {"latitude": 46.174640576447764, "longitude": 14.804130481638964},
    "Pomurska": {"latitude": 46.52351975291412, "longitude": 16.44175950632071},
    "Kozina": {"latitude": 45.60742223894982, "longitude": 13.927767896289717}
}

# Function to extract coordinates from traffic csv data files
def get_traffic_coordinates():
    # Get a list of all files in the 'processed_data_path' directory
    files = os.listdir(processed_data_path)
    coordinates_collection = pd.DataFrame(columns=['latitude', 'longitude'])

    for file in files:
        if file.endswith("_traffic_data.csv"):
            file_path = os.path.join(processed_data_path, file)

            df = pd.read_csv(file_path)
            coordinates = df[['latitude', 'longitude']]
            coordinates_collection = pd.concat([coordinates_collection, coordinates], ignore_index=True)

    return coordinates_collection

# Haversine formula to calculate distance between two points on the Earth
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

# Function to get the closest region based on latitude and longitude
def get_closest_region(lat, lon):
    min_distance = float('inf')
    closest_region = None

    for region, coords in cities_coordinates.items():
        distance = haversine(lat, lon, coords['latitude'], coords['longitude'])
        if distance < min_distance:
            min_distance = distance
            closest_region = region

    return closest_region


# Function to fetch weather data from Open-Meteo API
def fetch_weather_data(coordinates): 
    coordinates_lat = ""
    coordinates_lng = ""

    for _, coord in coordinates.iterrows():
        coordinates_lat += "," + str(coord['latitude']) + ","
        coordinates_lng += "," + str(coord['longitude']) + ","
    # Remove the last comma
    coordinates_lat = coordinates_lat.rstrip(',')
    coordinates_lng = coordinates_lng.rstrip(',')

    url = f"https://api.open-meteo.com/v1/forecast?latitude={coordinates_lat}&longitude={coordinates_lng}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,rain,snowfall,visibility&forecast_days=1&timezone=auto"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    
# Function for saving given data to a JSON file    
def save_to_json(data, filename):
    filepath = os.path.join(raw_data_path, filename)
    with open(filepath, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)  

def main():
    coordinates = get_traffic_coordinates()
    weather_data = fetch_weather_data(coordinates)
    
    # Add the region to the weather data
    for weather_instance in weather_data:
        weather_instance['region'] = get_closest_region(weather_instance['latitude'], weather_instance['longitude'])

    if weather_data:
        save_to_json(weather_data, "raw_weather_data.json")
        print("Weather data fetched and saved successfully!")
    else:
        print("Failed to fetch weather data.")


if __name__ == "__main__":
    main()