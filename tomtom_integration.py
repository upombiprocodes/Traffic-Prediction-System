import requests
import time

def fetch_real_time_traffic(api_key, lat, lon):
    """
    Fetches real-time traffic flow data from TomTom API.
    Returns a dictionary with speed and congestion info.
    """
    if not api_key:
        return None
        
    # TomTom Traffic Flow API endpoint
    # https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data
    base_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    
    params = {
        "key": api_key,
        "point": f"{lat},{lon}"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        flow_data = data.get("flowSegmentData", {})
        
        # Extract relevant metrics
        current_speed = flow_data.get("currentSpeed", 0) # km/h
        free_flow_speed = flow_data.get("freeFlowSpeed", 0) # km/h
        confidence = flow_data.get("confidence", 0)
        
        # Calculate congestion level (0 to 100%)
        # If current speed is near free flow, congestion is low.
        if free_flow_speed > 0:
            congestion = max(0, min(100, (1 - (current_speed / free_flow_speed)) * 100))
        else:
            congestion = 0
            
        return {
            "current_speed": int(current_speed),
            "free_flow_speed": int(free_flow_speed),
            "congestion_level": int(congestion),
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"TomTom API Error: {e}")
        return None

if __name__ == "__main__":
    # Test with a dummy key (will fail, but checks import)
    print(fetch_real_time_traffic("TEST_KEY", 40.7128, -74.0060))
