import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging
import json
import time
from datetime import datetime


# Configure logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("seat_monitor_api")

# This is the important line that gunicorn looks for:
app = Flask(__name__)
# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Path to the model file
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/seat_monitor_model.pkl')
THRESHOLDS_PATH = os.environ.get('THRESHOLDS_PATH', 'models/optimal_thresholds.txt')

# Global variables
model = None
optimal_thresholds = {}
sensor_data_history = []

# Load pre-trained model
def load_trained_model():
    global model, optimal_thresholds
    
    try:
        # Load the model
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully")
        
        # Load the optimal thresholds if available
        if os.path.exists(THRESHOLDS_PATH):
            logger.info(f"Loading thresholds from {THRESHOLDS_PATH}")
            with open(THRESHOLDS_PATH, 'r') as f:
                lines = f.readlines()
                
                # Parse thresholds from file
                for line in lines:
                    if line.startswith("Time:"):
                        parts = line.strip().split(", ")
                        if len(parts) == 2:
                            time_part = parts[0].replace("Time: ", "").replace(" seconds", "")
                            fsr_part = parts[1].replace("FSR: ", "")
                            try:
                                optimal_thresholds[int(time_part)] = int(float(fsr_part))
                            except ValueError:
                                continue
            
            logger.info(f"Loaded {len(optimal_thresholds)} threshold values")
        else:
            logger.warning(f"Thresholds file not found: {THRESHOLDS_PATH}")
            # Set default thresholds if file not found
            optimal_thresholds = {
                30: 200,   # 30 seconds
                60: 180,   # 1 minute
                120: 160,  # 2 minutes
                300: 140,  # 5 minutes
                600: 120   # 10 minutes
            }
            
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# Initialize model
model_loaded = load_trained_model()

@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    logger.info("Health check endpoint called")
    
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "thresholds_available": len(optimal_thresholds) > 0,
        "timestamp": time.time(),
        "api_version": "2.0"
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Make predictions based on FSR value and sitting time."""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    start_time = time.time()
    
    try:
        # Get the JSON data
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        
        # Validate input data
        if not data or 'fsr_value' not in data or 'sitting_time_seconds' not in data:
            logger.warning("Invalid request data - missing required fields")
            return jsonify({
                "error": "Missing required parameters: fsr_value and sitting_time_seconds are required"
            }), 400
        
        # Extract values
        fsr_value = float(data['fsr_value'])
        sitting_time_seconds = float(data['sitting_time_seconds'])
        
        logger.info(f"Making prediction with: FSR={fsr_value}, Time={sitting_time_seconds}s")
        
        # IMPORTANT: For high FSR values (900+), DO NOT vibrate until minimum time
        if fsr_value > 800:
            # Minimum sitting time required before vibration (30 seconds)
            min_time_for_vibration = 30
            
            if sitting_time_seconds < min_time_for_vibration:
                # Less than minimum time - no vibration regardless of pressure
                should_vibrate = False
                confidence = 0.9
                reason = "High pressure detected but sitting time too short yet"
            elif sitting_time_seconds >= 60:
                # Full minute of sitting with high pressure - definitely vibrate
                should_vibrate = True
                confidence = 0.95
                reason = "Extended high pressure sitting - break recommended"
            else:
                # Between min_time and full minute - consider vibration for very high pressures
                should_vibrate = fsr_value > 950
                confidence = 0.8
                reason = "High pressure and moderate sitting time - posture check suggested"
            
            return jsonify({
                "should_vibrate": should_vibrate,
                "confidence": confidence,
                "fsr_value": fsr_value,
                "sitting_time_seconds": sitting_time_seconds,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "reasoning": reason
            })

        
        # If model is available, use it for prediction
        if model is not None:
            # Create test sample
            test_sample = pd.DataFrame({
                'fsr': [fsr_value],
                'time': [sitting_time_seconds]
            })
            
            # Make prediction using the trained model
            prediction = bool(model.predict(test_sample)[0])
            
            # Get probability
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(test_sample)[0][1])
            else:
                probability = 0.7 if prediction else 0.3
                
            # Add basic reasoning based on the prediction
            if prediction:
                if sitting_time_seconds > 300:
                    reason = "Extended sitting time detected - break recommended"
                elif fsr_value > 500:
                    reason = "High pressure detected - consider adjusting posture"
                else:
                    reason = "Model predicts a break would be beneficial based on learned patterns"
            else:
                if sitting_time_seconds < 60:
                    reason = "Short sitting duration - no break needed yet"
                elif fsr_value < 150:
                    reason = "Low pressure indicates good posture"
                else:
                    reason = "Your sitting pattern indicates you're comfortable at the moment"
                    
            result = {
                "should_vibrate": prediction,
                "confidence": probability,
                "fsr_value": fsr_value,
                "sitting_time_seconds": sitting_time_seconds,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "reasoning": reason
            }
            
            logger.info(f"Prediction result: should_vibrate={prediction}, confidence={probability:.2f}")
            return jsonify(result)
        else:
            # Fallback if model not available
            logger.warning("Model not available, using fallback rules")
            
            # Simple rules based on time and pressure
            should_vibrate = (sitting_time_seconds > 120 and fsr_value > 200) or sitting_time_seconds > 300
            confidence = 0.6
            reason = "Using basic rules (model not available): " + (
                "Extended sitting detected" if should_vibrate else "Continue sitting"
            )
            
            return jsonify({
                "should_vibrate": should_vibrate,
                "confidence": confidence,
                "fsr_value": fsr_value,
                "sitting_time_seconds": sitting_time_seconds,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "reasoning": reason
            })
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "should_vibrate": False,
            "confidence": 0.0
        }), 500

@app.route('/recommend-settings', methods=['GET', 'OPTIONS'])
def recommend_settings():
    """Recommend optimal settings based on sitting time."""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        # Get sitting time from query parameters
        sitting_time_minutes = request.args.get('sitting_time_minutes', default=1, type=int)
        sitting_time_seconds = sitting_time_minutes * 60
        
        logger.info(f"Recommendation requested for {sitting_time_minutes} minute intervals")
        
        # Check sensor history for high FSR values
        high_pressure = False
        avg_fsr = 0
        
        if sensor_data_history:
            fsr_values = [entry['fsr'] for entry in sensor_data_history]
            avg_fsr = sum(fsr_values) / len(fsr_values)
            high_pressure = avg_fsr > 700
        
        # For high pressure users (~900), recommend special settings
        if high_pressure:
            return jsonify({
                "recommended_settings": {
                    "threshold": 180,  # Higher threshold for high pressure users
                    "sitting_time_seconds": 30,  # 30 seconds (shorter interval)
                    "vibration_duration_seconds": 3  # Longer vibration
                },
                "user_friendly": {
                    "pressure_sensitivity": "high",
                    "sitting_time_minutes": 0,
                    "sitting_time_seconds": 30,
                    "vibration_duration": 3
                },
                "message": "Based on your high pressure readings, we recommend more frequent breaks.",
                "reasoning": f"Your average pressure of {round(avg_fsr)} is quite high, suggesting potential posture issues."
            })
            
        # Use optimal thresholds from the trained model if available
        if optimal_thresholds:
            # Find the closest threshold time
            closest_time = min(optimal_thresholds.keys(), key=lambda x: abs(x - sitting_time_seconds))
            recommended_fsr = optimal_thresholds.get(closest_time, 150)
            
            # Adjust vibration duration based on sitting time
            if sitting_time_seconds <= 60:
                vibration_duration = 2  # 2 seconds
            elif sitting_time_seconds <= 300:
                vibration_duration = 3  # 3 seconds
            else:
                vibration_duration = 4  # 4 seconds
                
            # Generate user-friendly sensitivity label
            fsr_sensitivity = "medium"
            if recommended_fsr < 120:
                fsr_sensitivity = "high"
            elif recommended_fsr > 200:
                fsr_sensitivity = "low"
                
            result = {
                "recommended_settings": {
                    "threshold": recommended_fsr,
                    "sitting_time_seconds": sitting_time_seconds,
                    "vibration_duration_seconds": vibration_duration
                },
                "user_friendly": {
                    "pressure_sensitivity": fsr_sensitivity,
                    "sitting_time_minutes": int(sitting_time_seconds / 60),
                    "sitting_time_seconds": sitting_time_seconds % 60,
                    "vibration_duration": vibration_duration
                },
                "message": f"These recommendations are based on our trained model using 2000 data points.",
                "reasoning": f"For {sitting_time_minutes} minute intervals, a pressure threshold of {recommended_fsr} is optimal."
            }
            
            logger.info(f"Sending recommendation: {result['recommended_settings']}")
            return jsonify(result)
        else:
            # Fallback if thresholds not available
            logger.warning("Thresholds not available, using fallback recommendations")
            
            # Simple recommendations based on sitting time
            if sitting_time_minutes <= 1:
                recommended_settings = {
                    "threshold": 180,
                    "sitting_time_seconds": 60,
                    "vibration_duration_seconds": 2
                }
                message = "Short interval setting for frequent reminders."
            elif sitting_time_minutes <= 5:
                recommended_settings = {
                    "threshold": 150,
                    "sitting_time_seconds": sitting_time_minutes * 60,
                    "vibration_duration_seconds": 3
                }
                message = "Medium interval setting for balanced breaks."
            else:
                recommended_settings = {
                    "threshold": 120,
                    "sitting_time_seconds": sitting_time_minutes * 60,
                    "vibration_duration_seconds": 4
                }
                message = "Longer interval setting for deeper work sessions."
                
            result = {
                "recommended_settings": recommended_settings,
                "user_friendly": {
                    "pressure_sensitivity": "medium",
                    "sitting_time_minutes": sitting_time_minutes,
                    "sitting_time_seconds": 0,
                    "vibration_duration": recommended_settings["vibration_duration_seconds"]
                },
                "message": message
            }
            
            logger.info(f"Sending fallback recommendation: {result['recommended_settings']}")
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({
            "error": f"Failed to generate recommendations: {str(e)}"
        }), 500

def _build_cors_preflight_response():
    """Build a response for CORS preflight requests"""
    response = Response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
    return response
if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting server on port {port}")
    logger.info(f"API endpoints available:")
    logger.info(f"  * /health - Check if API is healthy")
    logger.info(f"  * /predict - Make predictions based on FSR and sitting time")
    logger.info(f"  * /recommend-settings - Get recommended settings for a given sitting time")
    
    # Run the app with appropriate settings
    app.run(host='0.0.0.0', port=port)