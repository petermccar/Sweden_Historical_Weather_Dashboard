"""
Fetch historical weather data from Open-Meteo API for Swedish locations.
This script downloads 20 years of daily weather data (2005-2025) for 10 Swedish cities.
"""

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime
import os
import time

# Setup the Open-Meteo API client with cache and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Swedish locations with their coordinates
LOCATIONS = {
    "Kiruna": {"lat": 67.8558, "lon": 20.2253},
    "Luleå": {"lat": 65.5848, "lon": 22.1547},
    "Hemavan": {"lat": 65.8089, "lon": 15.0819},
    "Östersund": {"lat": 63.1792, "lon": 14.6357},
    "Sundsvall": {"lat": 62.3908, "lon": 17.3069},
    "Stockholm": {"lat": 59.3293, "lon": 18.0686},
    "Örebro": {"lat": 59.2753, "lon": 15.2134},
    "Norrköping": {"lat": 58.5877, "lon": 16.1924},
    "Göteborg": {"lat": 57.7089, "lon": 11.9746},
    "Malmö": {"lat": 55.6050, "lon": 13.0038}
}

# Date range
START_DATE = "2005-01-01"
END_DATE = "2025-12-31"

# API endpoint
URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_data_for_location(location_name, lat, lon):
    """
    Fetch weather data for a single location.
    
    Args:
        location_name: Name of the location
        lat: Latitude
        lon: Longitude
    
    Returns:
        pandas DataFrame with weather data
    """
    print(f"Fetching data for {location_name}...")
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "apparent_temperature_mean",
            "daylight_duration",
            "sunshine_duration",
            "rain_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant"
        ],
        "timezone": "Europe/Stockholm"
    }
    
    try:
        responses = openmeteo.weather_api(URL, params=params)
        response = responses[0]
        
        # Process daily data
        daily = response.Daily()
        
        # Create DataFrame
        weather_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "location": location_name,
            "latitude": lat,
            "longitude": lon,
            "temperature_max": daily.Variables(0).ValuesAsNumpy(),
            "temperature_min": daily.Variables(1).ValuesAsNumpy(),
            "temperature_mean": daily.Variables(2).ValuesAsNumpy(),
            "apparent_temperature_max": daily.Variables(3).ValuesAsNumpy(),
            "apparent_temperature_min": daily.Variables(4).ValuesAsNumpy(),
            "apparent_temperature_mean": daily.Variables(5).ValuesAsNumpy(),
            "daylight_duration": daily.Variables(6).ValuesAsNumpy(),
            "sunshine_duration": daily.Variables(7).ValuesAsNumpy(),
            "rain_sum": daily.Variables(8).ValuesAsNumpy(),
            "snowfall_sum": daily.Variables(9).ValuesAsNumpy(),
            "wind_speed_max": daily.Variables(10).ValuesAsNumpy(),
            "wind_gusts_max": daily.Variables(11).ValuesAsNumpy(),
            "wind_direction_dominant": daily.Variables(12).ValuesAsNumpy()
        }
        
        df = pd.DataFrame(weather_data)
        
        # Convert date to timezone-aware then remove timezone for cleaner storage
        df['date'] = df['date'].dt.tz_convert('Europe/Stockholm').dt.tz_localize(None)
        
        print(f"✓ Successfully fetched {len(df)} days of data for {location_name}")
        return df
        
    except Exception as e:
        print(f"✗ Error fetching data for {location_name}: {e}")
        return None


def fetch_all_weather_data():
    """
    Fetch weather data for all locations and combine into a single DataFrame.
    
    Returns:
        pandas DataFrame with all weather data
    """
    all_data = []
    
    print(f"\nFetching weather data from {START_DATE} to {END_DATE}")
    print(f"Locations: {len(LOCATIONS)}")
    print("-" * 60)
    
    for i, (location_name, coords) in enumerate(LOCATIONS.items()):
        df = fetch_weather_data_for_location(
            location_name, 
            coords["lat"], 
            coords["lon"]
        )
        if df is not None:
            all_data.append(df)
        
        # Add delay between requests to avoid rate limiting (except after the last location)
        if i < len(LOCATIONS) - 1:
            print("  Waiting 65 seconds to respect API rate limits...")
            time.sleep(65)  # Wait 65 seconds between requests
    
    if all_data:
        # Combine all DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        print("-" * 60)
        print(f"\n✓ Total records fetched: {len(combined_df):,}")
        print(f"✓ Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        return combined_df
    else:
        print("✗ No data was fetched successfully")
        return None


def save_weather_data(df, filename="swedish_weather_data.csv"):
    """
    Save weather data to CSV file.
    
    Args:
        df: pandas DataFrame with weather data
        filename: Output filename
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    filepath = os.path.join("data", filename)
    df.to_csv(filepath, index=False)
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n✓ Data saved to: {filepath}")
    print(f"✓ File size: {file_size_mb:.2f} MB")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Swedish Weather Data Fetcher")
    print("=" * 60)
    
    # Fetch all data
    weather_df = fetch_all_weather_data()
    
    if weather_df is not None:
        # Save to CSV
        save_weather_data(weather_df)
        
        # Display summary statistics
        print("\n" + "=" * 60)
        print("Data Summary")
        print("=" * 60)
        print(f"\nLocations included:")
        for location in weather_df['location'].unique():
            count = len(weather_df[weather_df['location'] == location])
            print(f"  • {location}: {count:,} days")
        
        print(f"\nTemperature range (°C):")
        print(f"  • Min: {weather_df['temperature_min'].min():.1f}°C")
        print(f"  • Max: {weather_df['temperature_max'].max():.1f}°C")
        
        print("\n✓ Data fetching complete!")
    else:
        print("\n✗ Data fetching failed")


if __name__ == "__main__":
    main()
