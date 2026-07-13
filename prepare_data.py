import numpy as np
import pandas as pd

def generate_flood_dataset():
    # Set seed for reproducibility
    np.random.seed(42)
    
    n_samples = 290 # 290 samples, 20% test split is 58 samples.
    
    # Generate independent variables
    annual_rainfall = np.random.normal(2000, 500, n_samples)
    # Ensure all values are positive
    annual_rainfall = np.clip(annual_rainfall, 800, 4000)
    
    # Seasonal rainfall is highly correlated with annual rainfall (usually ~60-70%)
    seasonal_rainfall = annual_rainfall * np.random.uniform(0.55, 0.75, n_samples) + np.random.normal(0, 100, n_samples)
    seasonal_rainfall = np.clip(seasonal_rainfall, 400, 3000)
    
    # Cloud visibility (km) - lower visibility means more clouds/rain
    cloud_visibility = np.random.normal(6.5, 2.0, n_samples)
    cloud_visibility = np.clip(cloud_visibility, 1.0, 15.0)
    
    # Temperature
    temperature = np.random.normal(27, 4, n_samples)
    temperature = np.clip(temperature, 15, 42)
    
    # Humidity
    humidity = np.random.normal(75, 10, n_samples)
    humidity = np.clip(humidity, 35, 100)
    
    # Cloud Cover (Categorical: Low, Medium, High)
    cloud_cover_probs = []
    for visibility in cloud_visibility:
        if visibility < 4.0:
            cloud_cover_probs.append([0.05, 0.15, 0.80]) # High probability of High cloud cover
        elif visibility < 8.0:
            cloud_cover_probs.append([0.10, 0.70, 0.20]) # High probability of Medium cloud cover
        else:
            cloud_cover_probs.append([0.80, 0.15, 0.05]) # High probability of Low cloud cover
            
    cloud_cover = [np.random.choice(["Low", "Medium", "High"], p=probs) for probs in cloud_cover_probs]
    
    # Let's define a scoring function for likelihood of a flood
    # High annual rainfall, high seasonal rainfall, low visibility, high humidity increase flood risk
    scaled_annual = (annual_rainfall - 2000) / 500
    scaled_seasonal = (seasonal_rainfall - 1300) / 400
    scaled_vis = (6.5 - cloud_visibility) / 2.0
    scaled_hum = (humidity - 75) / 10
    
    score = 0.4 * scaled_annual + 0.4 * scaled_seasonal + 0.1 * scaled_vis + 0.1 * scaled_hum
    
    # Base probability
    prob = 1 / (1 + np.exp(-score))
    
    # Add minor noise
    prob = np.clip(prob + np.random.normal(0, 0.05, n_samples), 0, 1)
    
    # Binary classification
    y_class = []
    for p in prob:
        if p > 0.5:
            y_class.append("Flood")
        else:
            y_class.append("No Flood")
            
    df = pd.DataFrame({
        "Annual Rainfall": np.round(annual_rainfall, 2),
        "Cloud Visibility": np.round(cloud_visibility, 2),
        "Seasonal Rainfall": np.round(seasonal_rainfall, 2),
        "Temperature": np.round(temperature, 2),
        "Humidity": np.round(humidity, 2),
        "Cloud Cover": cloud_cover,
        "class": y_class
    })
    
    # Save to Excel
    df.to_excel("flood_dataset.xlsx", index=False)
    print("Dataset generated successfully and saved to flood_dataset.xlsx")
    print(df["class"].value_counts())

if __name__ == "__main__":
    generate_flood_dataset()
