import requests
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')
raw_data_path = os.path.join(data_dir, 'raw')

    
# Function to fetch traffic data from goriva.si API
def fetch_fuel_data():
    url = "https://goriva.si/api/v1/search/?franchise=&name=&o=price_95&page=2&position=Ljubljana&radius="
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching fuel data: {e}")
        return None

# Function for saving given data to a JSON file
def save_to_json(data, filename):
    filepath = os.path.join(raw_data_path, filename)
    with open(filepath, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)  

def main():
    fuel_data = fetch_fuel_data()  

    if fuel_data:
        save_to_json(fuel_data, "raw_fuel_data.json")
        print("Fuel data fetched and saved successfully!")
    else:
        print("Failed to fetch fuel data.")


if __name__ == "__main__":
    main()