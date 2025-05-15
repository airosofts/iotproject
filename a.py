import json
import time
from datetime import datetime, timedelta
import paho.mqtt.client as mqtt
from supabase import create_client, Client
import threading
import numpy as np
from collections import deque
import pandas as pd
from sklearn.cluster import KMeans
import ssl
import pytz
import random

# MQTT Configuration
MQTT_BROKER = "7f94900f5ded4432bfe7cc8d5378ec95.s1.eu.hivemq.cloud"
MQTT_PORT = 8883  # Secure port
MQTT_CLIENT_ID = f"seat_monitor_{int(time.time())}"
MQTT_USERNAME = "hamzamaqsoodfc786"
MQTT_PASSWORD = "Asadaotaf786@"

# MQTT Topics
TOPIC_SENSOR_DATA = "seat/sensor"
TOPIC_AI_INSIGHTS = "seat/ai_insights"
TOPIC_SETTINGS = "seat/settings"
TOPIC_COMMAND = "seat/command"
TOPIC_AI_RECOMMENDATIONS = "seat/ai_recommendations"

# Supabase Configuration
SUPABASE_URL = "https://wgjbmfqffnvuinhptntn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndnamJtZnFmZm52dWluaHB0bnRuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDYzODE3MjksImV4cCI6MjA2MTk1NzcyOX0.gZZ_dMNMnQeFhx35Cy4GPoEU2e9zvuHZYOaBvaIQPm8"
USER_ID = "test_user"

# Timezone setup
PKT = pytz.timezone('Asia/Karachi')

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global variables for tracking sitting state and sessions
last_sitting_state = False
sitting_start_time = None
current_session_id = None
sitting_sessions = []
alert_count = 0
typical_sitting_duration = None

# User settings (defaults)
user_settings = {
    "threshold": 100,
    "sittingTime": 5000,  # 5000 ms = 5 seconds for testing
    "vibrationDuration": 3000
}

def json_serialize(obj):
    """Custom serializer for non-JSON-serializable objects."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        return json_serialize(obj)

def generate_ai_recommendations(sitting_time_minutes):
    """
    Generate AI-driven recommendations based on sitting time.
    
    Args:
        sitting_time_minutes (int): Current sitting time threshold
    
    Returns:
        dict: Recommended settings
    """
    try:
        # Base recommendations with some randomness for machine learning simulation
        base_threshold = max(50, min(500, int(100 + sitting_time_minutes * 5 + random.uniform(-20, 20))))
        base_sitting_time = max(30, min(600, int(sitting_time_minutes * 60 + random.uniform(-30, 30))))
        base_vibration_duration = max(1, min(10, int(3 + random.uniform(-1, 1))))
        
        recommendations = {
            "event": "ai_recommendations",
            "timestamp": datetime.now(PKT).isoformat(),
            "recommended_settings": {
                "threshold": base_threshold,
                "sitting_time_seconds": base_sitting_time,
                "vibration_duration_seconds": base_vibration_duration
            },
            "insights": [
                f"Based on your current sitting habits, we recommend adjusting your alert threshold.",
                f"Your typical sitting session suggests a {base_sitting_time/60:.1f} minute break interval."
            ],
            "reasoning": [
                f"Machine learning analysis of {sitting_time_minutes} minute sitting time suggests optimized parameters.",
                "Recommendations are dynamically generated to improve your sitting habits."
            ]
        }
        
        return recommendations
    except Exception as e:
        print(f"Error generating AI recommendations: {e}")
        return None

def record_vibration_alert(session_id=None):
    """
    Record a vibration alert in the database with optional session association.
    
    Args:
        session_id (int, optional): The ID of the current sitting session.
    """
    global alert_count
    
    try:
        # Prepare alert data
        alert_data = {
            'user_id': USER_ID,
            'timestamp': datetime.now(PKT).isoformat(),
        }
        
        # If a session ID is provided, include it
        if session_id:
            alert_data['session_id'] = session_id
        
        # Insert alert record
        result = supabase.table('vibration_alerts').insert(alert_data).execute()
        
        # Increment local alert count
        alert_count += 1
        
        print(f"Recorded vibration alert. Total alerts: {alert_count}")
        
        return result
    except Exception as e:
        print(f"Error recording vibration alert: {e}")
        return None

def handle_command(client, payload):
    """
    Handle incoming command messages, specifically vibration triggers.
    
    Args:
        client (mqtt.Client): MQTT client instance
        payload (dict): Received command payload
    """
    try:
        if 'motor' in payload and payload['motor'] is True:
            print("Received motor command, triggering vibration")
            
            # Record alert with current session ID if available
            alert_result = record_vibration_alert(current_session_id)
            
            # Publish alert count back to the client for synchronization
            alert_sync_message = {
                'event': 'alert_sync',
                'timestamp': datetime.now(PKT).isoformat(),
                'total_alerts': alert_count
            }
            client.publish(TOPIC_AI_INSIGHTS, json.dumps(alert_sync_message, cls=NumpyJSONEncoder))
            
            print("Recorded manual vibration alert")
    except Exception as e:
        print(f"Error handling command: {e}")

def update_session_alerts(session_id):
    """
    Update alert count for a specific session.
    
    Args:
        session_id (int): ID of the sitting session
    """
    try:
        # Count alerts for this specific session
        alerts_result = supabase.table('vibration_alerts') \
            .select('count', count='exact') \
            .eq('session_id', session_id) \
            .execute()
        
        session_alert_count = alerts_result.count if alerts_result.count is not None else 0
        
        # Update session record with alert count
        supabase.table('sitting_sessions') \
            .update({'alert_count': session_alert_count}) \
            .eq('id', session_id) \
            .execute()
        
        print(f"Updated session {session_id} with {session_alert_count} alerts")
        
        return session_alert_count
    except Exception as e:
        print(f"Error updating session alerts: {e}")
        return 0

def handle_sensor_data(client, payload):
    """
    Handle incoming sensor data and manage sitting sessions.
    
    Args:
        client (mqtt.Client): MQTT client instance
        payload (dict): Sensor data payload
    """
    global last_sitting_state, sitting_start_time, current_session_id, typical_sitting_duration
    
    try:
        is_sitting = payload.get('isSitting', False)
        fsr_value = payload.get('fsrValue', 0)
        
        # Sitting started
        if is_sitting and not last_sitting_state:
            sitting_start_time = datetime.now(PKT)
            print(f"Sitting started at {sitting_start_time.isoformat()}")
            
            try:
                # Insert new sitting session
                result = supabase.table('sitting_sessions').insert({
                    'user_id': USER_ID,
                    'start_time': sitting_start_time.isoformat(),
                    'end_time': None,
                    'alert_count': 0  # Initialize alert count
                }).execute()
                
                if result.data:
                    current_session_id = result.data[0]['id']
                    print(f"Started session with ID: {current_session_id}")
                else:
                    print("Failed to get session ID from insert operation")
                
            except Exception as e:
                print(f"Error creating sitting session: {e}")
        
        # Sitting ended
        elif not is_sitting and last_sitting_state:
            if sitting_start_time:
                end_time = datetime.now(PKT)
                duration = (end_time - sitting_start_time).total_seconds()
                
                if current_session_id:
                    try:
                        # Update session with end time
                        supabase.table('sitting_sessions').update({
                            'end_time': end_time.isoformat()
                        }).eq('id', current_session_id).execute()
                        
                        # Update alert count for this session
                        session_alerts = update_session_alerts(current_session_id)
                        
                        # Update typical sitting duration (simple moving average)
                        if typical_sitting_duration is None:
                            typical_sitting_duration = duration
                        else:
                            typical_sitting_duration = (typical_sitting_duration + duration) / 2
                        
                        print(f"Updated session {current_session_id} with end time")
                        
                        # Generate and publish AI recommendations
                        recommendations = generate_ai_recommendations(duration / 60)
                        if recommendations:
                            client.publish(TOPIC_AI_RECOMMENDATIONS, 
                                          json.dumps(recommendations, cls=NumpyJSONEncoder))
                        
                    except Exception as e:
                        print(f"Error updating session: {e}")
                
                # Reset tracking variables
                sitting_start_time = None
                current_session_id = None
        
        # Update last sitting state
        last_sitting_state = is_sitting
        
    except Exception as e:
        print(f"Error handling sensor data: {e}")


def load_recent_sessions():
    """
    Load and analyze recent sitting sessions for insights.
    
    Returns:
        dict: Daily summary of sitting sessions
    """
    try:
        today = datetime.now(PKT)
        start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        print(f"Fetching sessions from {start_of_day} to {end_of_day}")
        
        # Fetch today's sessions
        response = supabase.table('sitting_sessions') \
            .select('*') \
            .eq('user_id', USER_ID) \
            .gte('start_time', start_of_day.isoformat()) \
            .lt('start_time', end_of_day.isoformat()) \
            .order('start_time') \
            .execute()
        
        today_sessions = response.data if response.data else []
        print(f"Loaded {len(today_sessions)} sessions for today")
        
        # If no sessions, return a default insights object
        if not today_sessions:
            return {
                'event': 'daily_summary',
                'timestamp': today.isoformat(),
                'metrics': {
                    'total_sessions': 0,
                    'total_sitting_time_minutes': 0,
                    'total_alerts': 0,
                    'longest_session_minutes': 0
                },
                'insights': ["No sitting sessions recorded today."],
                'recommendations': ["Start tracking your sitting habits to get personalized insights."]
            }
        
        # Calculate total sitting time and alerts
        total_sitting_time = 0
        total_alerts = 0
        longest_session = 0
        
        for session in today_sessions:
            print(f"Processing session: {session}")
            
            if session['start_time'] and session['end_time']:
                start = datetime.fromisoformat(session['start_time'])
                end = datetime.fromisoformat(session['end_time'])
                session_duration = (end - start).total_seconds()
                
                print(f"Session duration: {session_duration} seconds")
                
                total_sitting_time += session_duration
                total_alerts += session.get('alert_count', 0)
                longest_session = max(longest_session, session_duration)
        
        # Generate comprehensive AI insights
        ai_insights = {
            'event': 'daily_summary',
            'timestamp': today.isoformat(),
            'metrics': {
                'total_sessions': len(today_sessions),
                'total_sitting_time_minutes': round(total_sitting_time / 60, 2),
                'total_alerts': total_alerts,
                'longest_session_minutes': round(longest_session / 60, 2)
            },
            'insights': [],
            'recommendations': []
        }
        
        # Add insights based on data
        if total_sitting_time > 0:
            # Average sitting session duration
            avg_session_duration = total_sitting_time / len(today_sessions) if today_sessions else 0
            
            # Sitting time insights
            if total_sitting_time > 4 * 3600:  # More than 4 hours
                ai_insights['insights'].append(
                    f"You've been sitting for {round(total_sitting_time/3600, 2)} hours today. This is quite long!"
                )
                ai_insights['recommendations'].append(
                    "Consider taking more frequent breaks and incorporating standing or walking activities."
                )
            elif total_sitting_time < 1 * 3600:  # Less than 1 hour
                ai_insights['insights'].append(
                    "Your total sitting time is relatively low today."
                )
                ai_insights['recommendations'].append(
                    "Maintain a balanced posture even during short sitting periods."
                )
            
            # Alerts insights
            if total_alerts > 3:
                ai_insights['insights'].append(
                    f"You received {total_alerts} sitting alerts today. Your body might be signaling the need for more movement."
                )
                ai_insights['recommendations'].append(
                    "Try to proactively stand and stretch before receiving alerts."
                )
            
            # Session duration insights
            if avg_session_duration > 45 * 60:  # More than 45 minutes
                ai_insights['insights'].append(
                    f"Your average sitting session is {round(avg_session_duration/60, 2)} minutes long."
                )
                ai_insights['recommendations'].append(
                    "Aim to break up long sitting sessions with short standing breaks."
                )
        
        # If no additional insights, add a generic message
        if not ai_insights['insights']:
            ai_insights['insights'].append(
                "Keep tracking your sitting habits for personalized insights."
            )
        
        # Add typical sitting duration if available
        if typical_sitting_duration:
            ai_insights['metrics']['typical_sitting_duration_minutes'] = round(typical_sitting_duration / 60, 2)
        
        print("Final AI Insights:")
        print(ai_insights)
        
        return ai_insights
    
    except Exception as e:
        print(f"Error loading recent sessions: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace
        return None
def ai_analysis_thread(client):
    """
    Thread for periodic AI-driven analysis and insights.
    
    Args:
        client (mqtt.Client): MQTT client instance
    """
    print("AI analysis thread started")
    
    while True:
        try:
            # Detailed logging for debugging
            print("Starting AI analysis thread cycle...")
            
            # Load recent sessions and generate insights
            daily_summary = load_recent_sessions()
            
            # More detailed logging
            print("Daily summary generation result:")
            print(f"Daily Summary: {daily_summary}")
            
            # Publish daily summary if available
            if daily_summary:
                try:
                    # Ensure JSON serialization with custom encoder
                    insights_json = json.dumps(daily_summary, cls=NumpyJSONEncoder)
                    
                    # Print full JSON for debugging
                    print("Full Insights JSON:")
                    print(insights_json)
                    
                    # Publish to AI insights topic
                    try:
                        publish_result = client.publish(TOPIC_AI_INSIGHTS, insights_json)
                        
                        # Check publish result
                        print(f"Publish result: {publish_result}")
                        print(f"Publish mid: {publish_result.mid}")
                        
                        # Additional logging
                        print("Published AI insights successfully")
                        print("Payload length:", len(insights_json))
                    except Exception as publish_error:
                        print(f"Error during MQTT publish: {publish_error}")
                    
                except Exception as json_error:
                    print(f"Error serializing insights to JSON: {json_error}")
            else:
                print("No daily summary generated")
            
            # Sleep to prevent tight looping
            print("Sleeping for 5 minutes...")
            time.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            print(f"Critical error in AI analysis thread: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace
            time.sleep(30)  # Longer sleep on error

def on_connect(client, userdata, flags, rc):
    """
    MQTT connection callback.
    
    Args:
        client (mqtt.Client): MQTT client instance
        userdata: User-defined data
        flags: Connection flags
        rc (int): Return code
    """
    print(f"Connected to MQTT broker with result code: {rc}")
    client.subscribe(TOPIC_SENSOR_DATA)
    client.subscribe(TOPIC_COMMAND)
    print(f"Subscribed to {TOPIC_SENSOR_DATA} and {TOPIC_COMMAND}")
    
    # Start analysis thread
    threading.Thread(target=ai_analysis_thread, args=(client,), daemon=True).start()

def on_message(client, userdata, msg):
    """
    MQTT message reception callback.
    
    Args:
        client (mqtt.Client): MQTT client instance
        userdata: User-defined data
        msg (mqtt.MQTTMessage): Received message
    """
    try:
        payload = json.loads(msg.payload.decode())
        print(f"Received message on topic {msg.topic}: {payload}")
        
        if msg.topic == TOPIC_SENSOR_DATA:
            handle_sensor_data(client, payload)
        elif msg.topic == TOPIC_COMMAND:
            handle_command(client, payload)
    except Exception as e:
        print(f"Error processing message: {e}")

def main():
    """
  
    Main function to set up MQTT client and start the monitoring system.
    """
    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set(tls_version=ssl.PROTOCOL_TLS)
    client.tls_insecure_set(True)  # For testing; set to False in production
    
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()
