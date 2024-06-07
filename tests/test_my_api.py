import requests
import json
import random
from datetime import datetime, timedelta

# Define the base URL for the API
base_url = "http://localhost:3001"

# Test data for /traffic/predict/ endpoint
def generate_random_predict_data():
    today = datetime.today().date()
    random_date = today + timedelta(days=random.randint(0, 13))
    date_str = random_date.strftime("%Y-%m-%d")
    hour = f"{random.randint(0, 23):02}:00"

    predict_data = {
        "data": [
            {
                "region": "Postojna",
                "date": date_str,
                "hour": hour
            }
        ]
    }
    return predict_data

# Test data for /traffic/evaluation/ endpoint
evaluation_data = {
    "data": [
        {
            "region": "Postojna"
        }
    ]
}

# Function to test /traffic/predict/ endpoint
def test_predict():
    url = f"{base_url}/traffic/predict/"
    headers = {
        "Content-Type": "application/json"
    }
    predict_data = generate_random_predict_data()
    response = requests.post(url, headers=headers, data=json.dumps(predict_data))
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    #print("Response JSON:")
    #print(response.json())

# Function to test /traffic/evaluation/ endpoint
def test_evaluate():
    url = f"{base_url}/traffic/evaluation/"
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(evaluation_data))
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 200
    #print("Response JSON:")
    #print(response.json())

if __name__ == "__main__":
    print("Testing /traffic/predict/ endpoint ...")
    test_predict()
    print("\nTesting /traffic/evaluation/ endpoint ...")
    test_evaluate()
