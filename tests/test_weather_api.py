import requests

def test_weather_api():
    # Define a set of dummy coordinates for testing
    dummy_latitude = 46.0706  # Latitude for Ljubljana
    dummy_longitude = 14.5778  # Longitude for Ljubljana
    
    # Construct the API URL with dummy coordinates
    url = f"https://api.open-meteo.com/v1/forecast?latitude={dummy_latitude}&longitude={dummy_longitude}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,rain,snowfall,visibility&forecast_days=1&timezone=auto"

    try:
        # Send a GET request to the API
        response = requests.get(url)
        
        assert response.status_code == 200
        print("Test passed: Status code 200 received - Weather API is active.")
    except AssertionError:
        print("Test failed: Unexpected response format or status code.")
    except Exception as e:
        print("Test failed with exception:", e)

if __name__ == "__main__":
    test_weather_api()