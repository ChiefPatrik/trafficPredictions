import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Define constants for the APIs
TOKEN_URL = "https://b2b.nap.si/uc/user/token"
TRAFFIC_DATA_URL = "https://b2b.nap.si/data/b2b.counters.geojson.en_US"

# Replace these with valid test credentials or environment variable values
TEST_USERNAME = os.getenv('NAP_USERNAME')  # Replace with a valid username
TEST_PASSWORD = os.getenv('NAP_PASSWORD')  # Replace with a valid password


def get_tokens():
    """Test the token-fetching functionality."""
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {
        "grant_type": "password",
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD
    }
    print("Testing token endpoint...")

    try:
        response = requests.post(TOKEN_URL, headers=headers, data=payload)
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # print("Access Token:", data.get("access_token"))
            # print("Refresh Token:", data.get("refresh_token"))
            return data.get("access_token")  # Return the token for further testing
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Exception occurred: {e}")

    return None


def fetch_traffic_data(access_token):
    """Test the traffic data-fetching functionality."""
    headers = {
        "Authorization": f"bearer {access_token}"
    }
    print("Testing traffic data endpoint...")

    try:
        response = requests.get(TRAFFIC_DATA_URL, headers=headers)
        assert response.status_code == 200
        print("Test passed: Status code 200 received - Traffic API is active.")
    except AssertionError:
        print("Test failed: Unexpected response format or status code.")
    except Exception as e:
        print("Test failed with exception:", e)


def test_traffic_api():
    # Test getting tokens
    access_token = get_tokens()

    if access_token:
        # Test fetching traffic data
        fetch_traffic_data(access_token)
    else:
        print("Skipping traffic data test due to missing access token.")

if __name__ == "__main__":
    test_traffic_api()
