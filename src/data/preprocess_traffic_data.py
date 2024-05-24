import json
import csv
import os
from datetime import datetime
from pyproj import Transformer

# Initialize the transformer for converting coordinates
transformer = Transformer.from_crs("EPSG:3912", "EPSG:4326")

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')

raw_data_path = os.path.join(data_dir, 'raw')
processed_data_path = os.path.join(data_dir, 'processed')


def read_json(path, filename):
    filepath = os.path.join(path, filename)
    with open(filepath, mode='r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_traffic_data(traffic_point):
    fieldnames = ['datetime',
                  'region',
                  'num_of_cars',
                  'avg_speed',
                  'latitude',
                  'longitude'
                  ]
    region = traffic_point['properties']['stevci_regija'].replace('ACB ', '').replace(' ', '').replace('.', '_')
    filename = f"{region}_traffic_data.csv"
    filepath = os.path.join(processed_data_path, filename)

    coordinates = traffic_point['geometry']['coordinates']
    latitude, longitude = transformer.transform(coordinates[0], coordinates[1])
    # print(f"Coordinates BEFORE: {coordinates[0]}, {coordinates[1]}")
    # print(f"Coordinates AFTER: {latitude}, {longitude}")
    
    # Check if file exists and is non-empty
    file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0

    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # Write header only if file is new/empty
        writer.writerow({
            'datetime': traffic_point['properties']['updated'],
            'region': region,
            'num_of_cars': traffic_point['properties']['stevci_stev'],
            'avg_speed': traffic_point['properties']['stevci_hit'],
            'latitude': latitude,
            'longitude': longitude
        })

def process_data(traffic_data):
    region_indexes = { 11, # Ljubljana
                       20, # Vransko
                       25, # Postojna
                       45, # Kozina
                       86, # Maribor
                       57, # Sl. Konjice
                       700 # Pomurska
                     }     
    for index in region_indexes:
        traffic_point = traffic_data["features"][index]
        save_traffic_data(traffic_point)
    print("Traffic data preprocessed and saved to CSV files successfully!")


def main():
    traffic_json_filename = "raw_traffic_data.json"
    traffic_data = read_json(raw_data_path, traffic_json_filename)
    process_data(traffic_data)

if __name__ == "__main__":
    main()