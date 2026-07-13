from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the saved model and scaler
model_path = 'floods.save'
scaler_path = 'transform.save'

model = None
scaler = None

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and Scaler loaded successfully.")
else:
    print("Warning: Model or Scaler file not found. Run train_models.py first.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Predict', methods=['GET'])
def predict_page():
    return render_template('index.html')

@app.route('/chance')
def chance():
    prob = request.args.get('prob', '94.0')
    annual = request.args.get('annual', 'N/A')
    seasonal = request.args.get('seasonal', 'N/A')
    vis = request.args.get('vis', 'N/A')
    temp = request.args.get('temp', 'N/A')
    hum = request.args.get('hum', 'N/A')
    cc = request.args.get('cc', 'N/A')
    return render_template('chance.html', prob=prob, annual=annual, seasonal=seasonal, vis=vis, temp=temp, hum=hum, cc=cc)

@app.route('/no_chance')
def no_chance():
    prob = request.args.get('prob', '94.0')
    annual = request.args.get('annual', 'N/A')
    seasonal = request.args.get('seasonal', 'N/A')
    vis = request.args.get('vis', 'N/A')
    temp = request.args.get('temp', 'N/A')
    hum = request.args.get('hum', 'N/A')
    cc = request.args.get('cc', 'N/A')
    return render_template('no_chance.html', prob=prob, annual=annual, seasonal=seasonal, vis=vis, temp=temp, hum=hum, cc=cc)

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        else:
            return "Model not trained yet. Please train the model.", 500
            
    try:
        annual_rainfall = float(request.form.get('annual_rainfall'))
        cloud_visibility = float(request.form.get('cloud_visibility'))
        seasonal_rainfall = float(request.form.get('seasonal_rainfall'))
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        cloud_cover_str = request.form.get('cloud_cover')
        
        cloud_cover_map = {"Low": 0, "Medium": 1, "High": 2}
        cloud_cover = cloud_cover_map.get(cloud_cover_str, 0)
        
        input_data = pd.DataFrame([{
            "Annual Rainfall": annual_rainfall,
            "Cloud Visibility": cloud_visibility,
            "Seasonal Rainfall": seasonal_rainfall,
            "Temperature": temperature,
            "Humidity": humidity,
            "Cloud Cover": cloud_cover
        }])
        
        scaled_features = scaler.transform(input_data)
        prediction = model.predict(scaled_features)[0]
        
        try:
            prob = model.predict_proba(scaled_features)[0]
            flood_prob = float(prob[0]) # probability of Flood
        except Exception:
            flood_prob = 1.0 if prediction == 0 else 0.0
            
        prob_val = round(flood_prob * 100, 1)
        
        if prediction == 0:
            return redirect(url_for('chance', prob=prob_val, annual=annual_rainfall, seasonal=seasonal_rainfall, vis=cloud_visibility, temp=temperature, hum=humidity, cc=cloud_cover_str))
        else:
            # For no chance, show the complement probability (safety percentage)
            safe_prob = round((1.0 - flood_prob) * 100, 1)
            return redirect(url_for('no_chance', prob=safe_prob, annual=annual_rainfall, seasonal=seasonal_rainfall, vis=cloud_visibility, temp=temperature, hum=humidity, cc=cloud_cover_str))
            
    except Exception as e:
        return f"Error during prediction: {str(e)}", 400

# API endpoint for React frontend integration
@app.route('/api/predict', methods=['POST'])
def api_predict():
    global model, scaler
    if model is None or scaler is None:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        else:
            return jsonify({"error": "Model not trained"}), 500
            
    try:
        data = request.json
        annual_rainfall = float(data.get('annual_rainfall'))
        cloud_visibility = float(data.get('cloud_visibility'))
        seasonal_rainfall = float(data.get('seasonal_rainfall'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        cloud_cover_str = data.get('cloud_cover')
        
        cloud_cover_map = {"Low": 0, "Medium": 1, "High": 2}
        cloud_cover = cloud_cover_map.get(cloud_cover_str, 0)
        
        input_data = pd.DataFrame([{
            "Annual Rainfall": annual_rainfall,
            "Cloud Visibility": cloud_visibility,
            "Seasonal Rainfall": seasonal_rainfall,
            "Temperature": temperature,
            "Humidity": humidity,
            "Cloud Cover": cloud_cover
        }])
        
        scaled_features = scaler.transform(input_data)
        prediction = int(model.predict(scaled_features)[0])
        
        # If prediction is 0, it is 'Flood'. If 1, it is 'No Flood'.
        # Let's calculate probability if XGBoost supports it.
        try:
            prob = model.predict_proba(scaled_features)[0]
            # prob[0] is probability of Flood (class 0)
            flood_prob = float(prob[0])
        except:
            flood_prob = 1.0 if prediction == 0 else 0.0
            
        return jsonify({
            "prediction": "Flood" if prediction == 0 else "No Flood",
            "prediction_code": prediction,
            "flood_probability": flood_prob
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
