import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')
raw_data_path = os.path.join(data_dir, 'raw')


# Function to get the access and refresh tokens
def get_tokens():
    # Define the URL and headers
    url = "https://b2b.nap.si/uc/user/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # Define the payload with credentials
    payload = {
        "grant_type": "password",
        "username": os.getenv('NAP_USERNAME'),  # Username from environment variable
        "password": os.getenv('NAP_PASSWORD')   # Password from environment variable
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        return data['access_token'], data['refresh_token']   
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tokens: {e}")
        return None
    
# Function to fetch traffic data from NAP.si API
def fetch_traffic_data(access_token):
    url = f"https://b2b.nap.si/data/b2b.counters.geojson.en_US"
    headers = {
        "Authorization": f"bearer {access_token}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching traffic data: {e}")
        return None

# Function for saving given data to a JSON file
def save_to_json(data, filename):
    filepath = os.path.join(raw_data_path, filename)
    with open(filepath, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)  

def main():
    access_token, refresh_token = get_tokens() 
    traffic_data = fetch_traffic_data(access_token)
    
    if traffic_data:
        save_to_json(traffic_data, "raw_traffic_data.json")
        print("Traffic data fetched and saved successfully!")
    else:
        print("Failed to fetch traffic data.")


if __name__ == "__main__":
    main()