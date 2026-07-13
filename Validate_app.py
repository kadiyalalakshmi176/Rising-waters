import urllib.request
import json

def test_prediction():
    url = "http://127.0.0.1:5000/api/predict"
    # Scenario 1 parameters
    data = {
        "annual_rainfall": 2800.0,
        "cloud_visibility": 3.2,
        "seasonal_rainfall": 1800.0,
        "temperature": 26.5,
        "humidity": 88.0,
        "cloud_cover": "High"
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers=headers, 
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            print("API Test Successful!")
            print("Response:", json.dumps(res_data, indent=2))
    except Exception as e:
        print("API Test Failed (is the Flask server running on port 5000?):", str(e))

if __name__ == "__main__":
    test_prediction()
