from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pickle
import numpy as np
import logging
from datetime import datetime
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the model
model = None

def load_model():
    """Load the ML model from PKL file"""
    global model
    try:
        with open(Config.MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def get_weather_from_api(latitude, longitude):
    """
    Fetch fresh weather data from Open-Meteo API
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m,rain,snowfall,precipitation,surface_pressure,wind_speed_10m,cloud_cover,relative_humidity_2m'
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Open-Meteo API request failed with status code: {response.status_code}")

def calculate_season_from_date(date_string):
    """
    Calculate season from date (0=Winter, 1=Spring, 2=Summer, 3=Autumn)
    """
    try:
        date_obj = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        month = date_obj.month
        
        if month in [12, 1, 2]:
            return 0.0  # Winter
        elif month in [3, 4, 5]:
            return 1.0  # Spring
        elif month in [6, 7, 8]:
            return 2.0  # Summer
        else:
            return 3.0  # Autumn
    except:
        return 3.0  # Default to Autumn if date parsing fails

def get_ai_weather_data(latitude, longitude):
    """
    Main function to get AI model formatted weather data
    """
    # Fetch fresh data from API
    weather_data = get_weather_from_api(latitude, longitude)
    hourly = weather_data['hourly']
    
    # Calculate averages
    def calculate_average(data_array):
        return sum(data_array) / len(data_array)
    
    # Calculate temperature extremes
    temp_data = hourly['temperature_2m']
    temp_max = max(temp_data)
    temp_min = min(temp_data)
    
    # Calculate all averages
    averages = {
        'temperature_2m': calculate_average(temp_data),
        'rain': calculate_average(hourly['rain']),
        'snowfall': calculate_average(hourly['snowfall']),
        'precipitation': calculate_average(hourly['precipitation']),
        'surface_pressure': calculate_average(hourly['surface_pressure']),
        'wind_speed_10m': calculate_average(hourly['wind_speed_10m']),
        'cloud_cover': calculate_average(hourly['cloud_cover']),
        'relative_humidity_2m': calculate_average(hourly['relative_humidity_2m'])
    }
    
    # Calculate season from the first date in the data
    first_date = hourly['time'][0]
    season = calculate_season_from_date(first_date)
    elevation = weather_data['elevation']
    
    # Format for AI model
    ai_input = {
        'temp': round(averages['temperature_2m'], 1),
        'precip': round(averages['precipitation'], 2),
        'rain': round(averages['rain'], 2),
        'snowfall': round(averages['snowfall'], 2),
        'humidity': round(averages['relative_humidity_2m'], 1),
        'windspeed': round(averages['wind_speed_10m'], 1),
        'pressure': round(averages['surface_pressure'], 1),
        'cloud_cover': round(averages['cloud_cover'], 1),
        'elevation': elevation,
        'season': season,
        'temp_max': round(temp_max, 1),
        'temp_min': round(temp_min, 1)
    }
    
    return ai_input

def make_flood_prediction(weather_features):
    """
    Make flood prediction using the loaded model
    """
    # Check if model is loaded
    if model is None:
        raise Exception("Model not loaded")

    # Extract features in exact order expected by the model
    features = [
        float(weather_features['temp_max']),
        float(weather_features['temp_min']), 
        float(weather_features['precip']),
        float(weather_features['rain']),
        float(weather_features['snowfall']),
        float(weather_features['humidity']),
        float(weather_features['windspeed']),
        float(weather_features['pressure']),
        float(weather_features['cloud_cover']),
        float(weather_features['elevation']),
        float(weather_features['season'])
    ]

    # Convert to numpy array and make prediction
    input_array = np.array([features])
    
    # Check if model has predict_proba (classification) or predict (regression)
    if hasattr(model, 'predict_proba'):
        # Classification model - get probability
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium" 
        else:
            risk_level = "Low"
            
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'message': 'Flood predicted' if prediction == 1 else 'No flood predicted'
        }
    else:
        # Regression model - direct prediction
        prediction = model.predict(input_array)[0]
        result = {
            'prediction': float(prediction),
            'message': 'Flood risk score'
        }

    logger.info(f"✅ Flood prediction made: {result}")
    return result

def calculate_flood_risk_equation(weather_features):
    """
    Heuristic flood-risk scoring (no ML). Returns risk_level, score and breakdown.
    Tune the max_points per factor below to adjust sensitivity.
    Expects weather_features keys: precip, rain, snowfall, humidity, windspeed,
    pressure, elevation, season, temp_max, temp_min
    """
    # safe conversions and defaults
    def f(k, default=0.0):
        try:
            return float(weather_features.get(k, default))
        except:
            return float(default)

    precip = f('precip')      # avg precipitation (units from API)
    rain = f('rain')
    snowfall = f('snowfall')
    humidity = f('humidity')
    wind = f('windspeed')
    pressure = f('pressure')
    elevation = f('elevation')
    season = int(f('season', 0))
    temp_max = f('temp_max')
    temp_min = f('temp_min')

    # Max points allocation (sum = 100)
    MAX = {
        'precip': 35,
        'rain': 25,
        'snow': 10,
        'humidity': 8,
        'pressure': 6,
        'wind': 5,
        'elevation': 6,
        'season': 5
    }

    breakdown = {}

    # Precipitation: scale to MAX['precip'] assuming ~50mm is very high
    breakdown['precip_points'] = min(MAX['precip'], (precip / 50.0) * MAX['precip'])

    # Rain intensity: scale assuming 20mm is strong
    breakdown['rain_points'] = min(MAX['rain'], (rain / 20.0) * MAX['rain'])

    # Snow melt risk: if snowfall significant and temp allows melting
    if snowfall > 5 and temp_max > 0:
        breakdown['snow_points'] = MAX['snow']
    else:
        breakdown['snow_points'] = 0.0

    # Humidity: only penalize when > 60%
    if humidity > 60:
        breakdown['humidity_points'] = min(MAX['humidity'], ((humidity - 60.0) / 40.0) * MAX['humidity'])
    else:
        breakdown['humidity_points'] = 0.0

    # Low pressure increases risk
    if pressure and pressure < 1000:
        breakdown['pressure_points'] = min(MAX['pressure'], ((1000.0 - pressure) / 50.0) * MAX['pressure'])
    else:
        breakdown['pressure_points'] = 0.0

    # Wind: high wind slightly increases risk
    breakdown['wind_points'] = min(MAX['wind'], (wind / 40.0) * MAX['wind'])

    # Elevation: lower elevation -> higher points (assume 0m worst, 1000m safe)
    if elevation is None:
        breakdown['elevation_points'] = 0.0
    else:
        elev_score = max(0.0, (1000.0 - elevation) / 1000.0)  # 0..1
        breakdown['elevation_points'] = min(MAX['elevation'], elev_score * MAX['elevation'])

    # Season: simple boost for spring (1)
    breakdown['season_points'] = MAX['season'] if season == 1 else 0.0

    # Sum and clamp
    total_score = sum(breakdown.values())
    risk_score = round(float(total_score), 1)

    # Determine level
    if risk_score >= 10:
        risk_level = "High"
        message = "High flood risk detected"
    elif risk_score >= 5:
        risk_level = "Medium"
        message = "Moderate flood risk"
    else:
        risk_level = "Low"
        message = "Low flood risk"

    # Log for debugging
    logger.info(f"[ManualRisk] averages: {weather_features}")
    logger.info(f"[ManualRisk] breakdown: {breakdown}, score={risk_score}, level={risk_level}")

    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'message': message,
        'breakdown': breakdown
    }

@app.route('/')
def home():
    """API documentation"""
    return jsonify({
        'message': 'Weather & Flood Prediction API',
        'endpoints': {
            '/api/weather': 'GET - Get weather data for coordinates',
            '/api/predict': 'GET - Get flood prediction for coordinates',
            '/api/health': 'GET - Health check',
            '/api/model-info': 'GET - Model information',
            '/api/manual-predict': 'POST - Manual prediction with custom data'
        },
        'usage': {
            'weather_data': '/api/weather?lat=49.0&lon=32.0',
            'prediction': '/api/predict?lat=49.0&lon=32.0'
        }
    })

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """
    Weather API endpoint - returns formatted weather data
    """
    try:
        # Get coordinates from query parameters
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if lat is None or lon is None:
            return jsonify({
                'error': 'Missing parameters',
                'message': 'Please provide both lat and lon parameters',
                'example': '/api/weather?lat=49.0&lon=32.0'
            }), 400
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({
                'error': 'Invalid coordinates',
                'message': 'Latitude must be between -90 and 90, Longitude between -180 and 180'
            }), 400
        
        # Get AI model formatted data
        ai_weather_data = get_ai_weather_data(lat, lon)
        
        return jsonify(ai_weather_data)
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid input',
            'message': 'Please provide valid numeric coordinates'
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['GET'])
def get_flood_prediction():
    """
    Flood prediction API endpoint - returns averaged weather data and
    a risk result computed by the manual equation (no ML model used).
    """
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)

        if lat is None or lon is None:
            return jsonify({
                'error': 'Missing parameters',
                'message': 'Please provide both lat and lon parameters',
                'example': '/api/predict?lat=49.0&lon=32.0'
            }), 400

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({
                'error': 'Invalid coordinates',
                'message': 'Latitude must be between -90 and 90, Longitude between -180 and 180'
            }), 400

        # Get averaged weather data
        weather_averages = get_ai_weather_data(lat, lon)

        # Compute risk using the manual equation (no model)
        risk_result = calculate_flood_risk_equation(weather_averages)

        response = {
            'status': 'success',
            'coordinates': {'latitude': lat, 'longitude': lon},
            'weather_averages': weather_averages,
            'risk_result': risk_result,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Prediction error (manual equation): {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/manual-predict', methods=['POST'])
def manual_predict():
    """
    Manual prediction endpoint - uses equation instead of model
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Use equation instead of model
        prediction = calculate_flood_risk_equation(data)
        
        logger.info(f"✅ Manual prediction made: {prediction}")
        return jsonify({
            'status': 'success',
            'input_data': data,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Manual prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy', 
        'message': 'Weather & Flood Prediction API is running',
        'model_status': model_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    model_type = type(model).__name__
    model_features = getattr(model, 'n_features_in_', 'Unknown')
    
    return jsonify({
        'model_type': model_type,
        'features_count': model_features,
        'input_format': {
            'temp_max': 'float',
            'temp_min': 'float', 
            'precip': 'float',
            'rain': 'float',
            'snowfall': 'float',
            'humidity': 'float',
            'windspeed': 'float',
            'pressure': 'float',
            'cloud_cover': 'float',
            'elevation': 'float',
            'season': 'float (0-3)'
        }
    })

# Load model when application starts
load_model()

if __name__ == '__main__':
    # Check if model loaded successfully
    if model is not None:
        logger.info("🚀 Starting Weather & Flood Prediction API...")
        app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
    else:
        logger.error("❌ Failed to load model. Cannot start API.")