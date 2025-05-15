import os
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

# MQTT Configuration from environment variables
MQTT_BROKER = os.environ.get('MQTT_BROKER', '7f94900f5ded4432bfe7cc8d5378ec95.s1.eu.hivemq.cloud')
MQTT_PORT = int(os.environ.get('MQTT_PORT', 8883))  # Secure port
MQTT_CLIENT_ID = f"seat_monitor_{int(time.time())}"
MQTT_USERNAME = os.environ.get('MQTT_USERNAME', 'hamzamaqsoodfc786')
MQTT_PASSWORD = os.environ.get('MQTT_PASSWORD', 'Asadaotaf786@')

# MQTT Topics
TOPIC_SENSOR_DATA = "seat/sensor"
TOPIC_AI_INSIGHTS = "seat/ai_insights"
TOPIC_SETTINGS = "seat/settings"
TOPIC_COMMAND = "seat/command"

# Supabase Configuration from environment variables
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://wgjbmfqffnvuinhptntn.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndnamJtZnFmZm52dWluaHB0bnRuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDYzODE3MjksImV4cCI6MjA2MTk1NzcyOX0.gZZ_dMNMnQeFhx35Cy4GPoEU2e9zvuHZYOaBvaIQPm8')
USER_ID = os.environ.get('USER_ID', 'test_user')

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
USER_ID = "test_user"

# Initialize Supabase client

# Tracking variables
last_sitting_state = False
sitting_start_time = None
current_session_id = None
sitting_sessions = []
today_sessions = []
weekly_sessions = []

# AI Analysis variables
daily_sitting_times = []
sitting_pattern_clusters = None
typical_sitting_duration = None
last_analysis_time = 0
last_daily_analysis_time = 0
last_weekly_analysis_time = 0

# User settings (defaults)
user_settings = {
    "threshold": 100,
    "sittingTime": 5000,
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

last_settings_check = 0
SETTINGS_CHECK_INTERVAL = 5
DAILY_ANALYSIS_INTERVAL = 30
WEEKLY_ANALYSIS_INTERVAL = 60

def load_user_settings():
    global user_settings, last_settings_check
    try:
        response = supabase.table('user_settings').select('*').eq('user_id', USER_ID).execute()
        if response.data and len(response.data) > 0:
            data = response.data[0]
            new_settings = {
                "threshold": data['threshold'],
                "sittingTime": data['sitting_time'],
                "vibrationDuration": data['vibration_duration']
            }
            if new_settings != user_settings:
                print(f"Settings changed: {user_settings} -> {new_settings}")
                user_settings = new_settings
                return True
            else:
                print("Settings unchanged")
                return False
        else:
            print("No user settings found, using defaults")
            return False
    except Exception as e:
        print(f"Error loading user settings: {e}")
        return False
    finally:
        last_settings_check = time.time()

def publish_settings_to_device(client):
    try:
        client.publish(TOPIC_SETTINGS, json.dumps(user_settings, cls=NumpyJSONEncoder))
        print(f"Published settings to device: {user_settings}")
        return True
    except Exception as e:
        print(f"Error publishing settings: {e}")
        return False

def load_recent_sessions():
    global today_sessions, weekly_sessions
    try:
        today = datetime.now()
        start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        start_of_week = start_of_day - timedelta(days=7)
        
        response = supabase.table('sitting_sessions') \
            .select('*') \
            .eq('user_id', USER_ID) \
            .gte('start_time', start_of_day.isoformat()) \
            .lt('start_time', end_of_day.isoformat()) \
            .order('start_time') \
            .execute()
        today_sessions = response.data if response.data else []
        
        response = supabase.table('sitting_sessions') \
            .select('*') \
            .eq('user_id', USER_ID) \
            .gte('start_time', start_of_week.isoformat()) \
            .lt('start_time', end_of_day.isoformat()) \
            .order('start_time') \
            .execute()
        weekly_sessions = response.data if response.data else []
        
        print(f"Loaded {len(today_sessions)} sessions for today and {len(weekly_sessions)} for the week")
        return True
    except Exception as e:
        print(f"Error loading recent sessions: {e}")
        return False

def analyze_sitting_patterns():
    global sitting_pattern_clusters, typical_sitting_duration
    load_recent_sessions()
    
    if len(weekly_sessions) < 3:
        print("Not enough data for pattern analysis")
        return None
    
    try:
        durations = []
        for session in weekly_sessions:
            if session['end_time']:
                start = datetime.fromisoformat(session['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(session['end_time'].replace('Z', '+00:00'))
                duration_seconds = (end - start).total_seconds()
                durations.append(duration_seconds)
        
        if len(durations) < 3:
            return None
        
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        min_duration = np.min(durations)
        total_sitting_time = sum(durations)
        
        if len(durations) >= 5:
            X = np.array(durations).reshape(-1, 1)
            kmeans = KMeans(n_clusters=min(3, len(durations)), random_state=0).fit(X)
            clusters = kmeans.predict(X)
            sitting_pattern_clusters = kmeans
            unique, counts = np.unique(clusters, return_counts=True)
            common_cluster = unique[np.argmax(counts)]
            cluster_durations = [d for i, d in enumerate(durations) if clusters[i] == common_cluster]
            typical_sitting_duration = np.mean(cluster_durations)
            pattern_insight = f"Your typical sitting session lasts around {int(typical_sitting_duration/60)} minutes."
        else:
            pattern_insight = f"Your average sitting time is {int(avg_duration/60)} minutes."
            typical_sitting_duration = avg_duration
        
        times_of_day = []
        for session in weekly_sessions:
            if session['start_time']:
                start = datetime.fromisoformat(session['start_time'].replace('Z', '+00:00'))
                times_of_day.append(start.hour)
        
        if times_of_day:
            hour_counts = pd.Series(times_of_day).value_counts()
            peak_hour = hour_counts.idxmax() if not hour_counts.empty else None
            time_insight = f"You tend to sit most frequently around {peak_hour}:00." if peak_hour is not None else "No clear pattern in your sitting times yet."
        else:
            time_insight = "No sitting time data available yet."
        
        recommendations = []
        threshold_minutes = user_settings['sittingTime'] / 60000
        if typical_sitting_duration / 60 > threshold_minutes:
            recommendations.append(f"Your typical sitting sessions ({int(typical_sitting_duration/60)} min) exceed your alert threshold ({threshold_minutes} min). Consider taking more frequent breaks.")
        if total_sitting_time > 6 * 3600:
            recommendations.append("You've been sitting for over 6 hours in the past week. Try incorporating more standing or walking activities.")
        
        days_of_week = []
        for session in weekly_sessions:
            if session['start_time']:
                start = datetime.fromisoformat(session['start_time'].replace('Z', '+00:00'))
                days_of_week.append(start.weekday())
        
        if days_of_week:
            day_counts = pd.Series(days_of_week).value_counts()
            busiest_day = day_counts.idxmax() if not day_counts.empty else None
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if busiest_day is not None and len(day_counts) > 1 and day_counts[busiest_day] > 1.5 * day_counts.mean():
                recommendations.append(f"You sit significantly more on {day_names[busiest_day]}s. Consider scheduling more movement breaks on this day.")
        
        analysis = {
            "event": "ai_analysis",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "avg_duration_minutes": round(avg_duration / 60, 1),
                "max_duration_minutes": round(max_duration / 60, 1),
                "min_duration_minutes": round(min_duration / 60, 1),
                "total_sitting_hours": round(total_sitting_time / 3600, 1),
                "session_count": len(durations)
            },
            "insights": [
                pattern_insight,
                time_insight
            ],
            "recommendations": recommendations
        }
        
        if peak_hour is not None:
            analysis["metrics"]["peak_hour"] = peak_hour
        
        return analysis
    except Exception as e:
        print(f"Error analyzing sitting patterns: {e}")
        return None

def ai_analysis_thread(client):
    global last_analysis_time, last_daily_analysis_time, last_weekly_analysis_time
    print("AI analysis thread started")
    
    while True:
        try:
            current_time = time.time()
            if current_time - last_weekly_analysis_time >= 60:
                print("Running quick AI analysis")
                load_recent_sessionspitfalls = True
                load_recent_sessions()
                last_analysis_time = current_time
            
            if current_time - last_daily_analysis_time >= DAILY_ANALYSIS_INTERVAL:
                print("Running daily AI analysis")
                analysis = analyze_sitting_patterns()
                if analysis:
                    client.publish(TOPIC_AI_INSIGHTS, json.dumps(analysis, cls=NumpyJSONEncoder))
                last_daily_analysis_time = current_time
            
            if current_time - last_weekly_analysis_time >= WEEKLY_ANALYSIS_INTERVAL:
                print("Running weekly AI analysis")
                load_recent_sessions()
                weekly_report = {
                    "event": "weekly_report",
                    "timestamp": datetime.now().isoformat(),
                    "session_count": len(weekly_sessions),
                    "recommendations": [
                        "Based on your sitting patterns this week, we recommend setting a 30-minute timer for standing breaks.",
                        "Consider using the pomodoro technique: 25 minutes of sitting followed by a 5-minute standing break."
                    ]
                }
                client.publish(TOPIC_AI_INSIGHTS, json.dumps(weekly_report, cls=NumpyJSONEncoder))
                last_weekly_analysis_time = current_time
                
            time.sleep(5)
        except Exception as e:
            print(f"Error in AI analysis thread: {e}")
            time.sleep(30)

def settings_monitor_thread(client):
    global last_settings_check
    print("Settings monitor thread started")
    
    while True:
        try:
            if time.time() - last_settings_check >= SETTINGS_CHECK_INTERVAL:
                if load_user_settings():
                    publish_settings_to_device(client)
            time.sleep(1)
        except Exception as e:
            print(f"Error in settings monitor thread: {e}")
            time.sleep(5)

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code: {rc}")
    client.subscribe(TOPIC_SENSOR_DATA)
    client.subscribe(TOPIC_COMMAND)
    print(f"Subscribed to {TOPIC_SENSOR_DATA} and {TOPIC_COMMAND}")
    
    load_user_settings()
    publish_settings_to_device(client)
    load_recent_sessions()
    
    settings_thread = threading.Thread(target=settings_monitor_thread, args=(client,))
    settings_thread.daemon = True
    settings_thread.start()
    
    ai_thread = threading.Thread(target=ai_analysis_thread, args=(client,))
    ai_thread.daemon = True
    ai_thread.start()
    
    client.publish(TOPIC_AI_INSIGHTS, json.dumps({
        "status": "AI Monitor Connected",
        "timestamp": datetime.now().isoformat()
    }, cls=NumpyJSONEncoder))

def on_message(client, userdata, msg):
    global last_sitting_state, sitting_start_time, current_session_id
    try:
        payload = json.loads(msg.payload.decode())
        print(f"Received message on topic {msg.topic}: {payload}")
        if msg.topic == TOPIC_SENSOR_DATA:
            handle_sensor_data(client, payload)
        elif msg.topic == TOPIC_COMMAND:
            handle_command(client, payload)
    except Exception as e:
        print(f"Error processing message: {e}")

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
from datetime import datetime
import pytz
import json
from supabase import create_client, Client

  # Assuming global variables and imports from your script
PKT = pytz.timezone('Asia/Karachi')
USER_ID = "test_user"
TOPIC_AI_INSIGHTS = "seat/ai_insights"

  # Placeholder for global variables
last_sitting_state = False
sitting_start_time = None
current_session_id = None
typical_sitting_duration = None
user_settings = {"sittingTime": 5000}
sitting_sessions = []

class NumpyJSONEncoder(json.JSONEncoder):
      def default(self, obj):
          if isinstance(obj, datetime):
              return obj.isoformat()
          return json.JSONEncoder.default(self, obj)

def handle_sensor_data(client, payload):
      global last_sitting_state, sitting_start_time, current_session_id
      try:
          is_sitting = payload.get('isSitting', False)
          fsr_value = payload.get('fsrValue', 0)
          print(f"FSR Value: {fsr_value}, Sitting: {is_sitting}")
          
          if is_sitting and not last_sitting_state:
              sitting_start_time = datetime.now(PKT)
              print(f"Sitting started at {sitting_start_time.isoformat()}")
              result = supabase.table('sitting_sessions').insert({
                  'user_id': USER_ID,
                  'start_time': sitting_start_time.isoformat(),
                  'end_time': None
              }).execute()
              if result.data:
                  current_session_id = result.data[0]['id']
                  print(f"Started session with ID: {current_session_id}")
              else:
                  print("Failed to get session ID from insert operation")
              
              message = {
                  "event": "sitting_started",
                  "timestamp": sitting_start_time.isoformat(),
                  "fsrValue": fsr_value
              }
              if typical_sitting_duration:
                  message["context"] = {
                      "typical_duration_minutes": round(typical_sitting_duration / 60, 1),
                      "alert_threshold_minutes": round(user_settings['sittingTime'] / 60000, 1)
                  }
                  if typical_sitting_duration > user_settings['sittingTime']:
                      message["suggestion"] = f"Based on your patterns, you typically sit for {round(typical_sitting_duration / 60, 1)} minutes. Consider setting a timer for {round(user_settings['sittingTime'] / 60000, 1)} minutes to take breaks more frequently."
              client.publish(TOPIC_AI_INSIGHTS, json.dumps(message, cls=NumpyJSONEncoder))
          
          elif not is_sitting and last_sitting_state:
              if sitting_start_time is not None:
                  end_time = datetime.now(PKT)
                  duration = (end_time - sitting_start_time).total_seconds()
                  if duration < 0:
                      print(f"Negative duration detected: start={sitting_start_time}, end={end_time}")
                      return
                  if current_session_id:
                      supabase.table('sitting_sessions').update({
                          'end_time': end_time.isoformat()
                      }).eq('id', current_session_id).execute()
                      print(f"Updated session {current_session_id} with end time")
                  
                  sitting_sessions.append({
                      "start": sitting_start_time.isoformat(),
                      "end": end_time.isoformat(),
                      "duration_seconds": duration
                  })
                  print(f"Sitting ended. Duration: {duration:.1f} seconds")
                  
                  message = {
                      "event": "sitting_ended",
                      "start_time": sitting_start_time.isoformat(),
                      "end_time": end_time.isoformat(),
                      "duration_seconds": round(duration, 1),
                      "duration_minutes": round(duration / 60, 1)
                  }
                  if typical_sitting_duration:
                      if duration < typical_sitting_duration * 0.5:
                          message["comparison"] = "This was a shorter sitting session than your usual pattern."
                      elif duration > typical_sitting_duration * 1.5:
                          message["comparison"] = "This was a longer sitting session than your usual pattern."
                      else:
                          message["comparison"] = "This sitting session was typical for your patterns."
                      durations = [s['duration_seconds'] for s in sitting_sessions if 'duration_seconds' in s]
                      if durations:
                          percentile = sum(1 for d in durations if d < duration) / len(durations) * 100
                          message["percentile"] = round(percentile, 1)
                  client.publish(TOPIC_AI_INSIGHTS, json.dumps(message, cls=NumpyJSONEncoder))
                  
                  sitting_start_time = None
                  current_session_id = None
          
          last_sitting_state = is_sitting
      except Exception as e:
          print(f"Error handling sensor data: {e}")
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

def main():
    try:
        response = supabase.table('sitting_sessions').select('count', count='exact').execute()
        print(f"Supabase connection successful! Found {response.count} sitting session records.")
    except Exception as e:
        print(f"Supabase connection error: {e}")
        print("Please check your Supabase URL and key.")
        return

    client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set(tls_version=ssl.PROTOCOL_TLS)
    client.tls_insecure_set(True)  # For testing; set to False in production with proper CA cert
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print("Starting MQTT loop...")
        client.loop_forever()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        print("Disconnecting...")
        client.disconnect()

if __name__ == "__main__":
    main()
