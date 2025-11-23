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

def fetch_real_time_incidents(api_key, lat, lon, radius=5000):
    """
    Fetches traffic incidents from TomTom API within a radius (meters).
    """
    if not api_key:
        return []
        
    # Construct bounding box from radius (approximate)
    # 1 deg lat ~ 111km, 1 deg lon ~ 111km * cos(lat)
    lat_delta = radius / 111000
    lon_delta = radius / (111000 * abs(requests.utils.quote(str(lat)) and 1)) # Simplified
    
    # Better bbox calculation
    min_lat = lat - 0.05 # approx 5km
    max_lat = lat + 0.05
    min_lon = lon - 0.05
    max_lon = lon + 0.05
    
    base_url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    
    params = {
        "key": api_key,
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "fields": "{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,events{description},startTime,endTime}}}"
    }
    
    try:
        # Incident Details endpoint needs specific structure
        # https://developer.tomtom.com/traffic-api/documentation/traffic-incidents/incident-details
        url = f"{base_url}?key={api_key}&bbox={min_lon},{min_lat},{max_lon},{max_lat}&fields={{incidents{{type,geometry{{type,coordinates}},properties{{iconCategory,magnitudeOfDelay,events{{description}},startTime,endTime}}}}}}"
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        incidents = []
        for item in data.get("incidents", []):
            props = item.get("properties", {})
            coords = item.get("geometry", {}).get("coordinates", [])
            
            # Extract description
            desc = "Traffic Incident"
            if "events" in props and props["events"]:
                desc = props["events"][0].get("description", desc)
                
            incidents.append({
                "type": props.get("iconCategory", "Unknown"),
                "severity": "High" if props.get("magnitudeOfDelay", 0) > 2 else "Moderate",
                "description": desc,
                "lat": coords[0][1] if coords else lat, # Simple centroid
                "lon": coords[0][0] if coords else lon
            })
            
        return incidents

    except Exception as e:
        print(f"TomTom Incident API Error: {e}")
        return None

if __name__ == "__main__":
    # Test with a dummy key (will fail, but checks import)
    print(fetch_real_time_traffic("TEST_KEY", 40.7128, -74.0060))
