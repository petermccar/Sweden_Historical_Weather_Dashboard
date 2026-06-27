# 🇸🇪 Sweden Historical Weather Dashboard

An interactive dashboard visualizing 20 years of historical weather data for 10 Swedish cities, built with Python and Streamlit.

## Features

The dashboard contains four interactive tabs, each with a clickable map of Sweden for location selection:

### 📊 Tab 1: Metric Ranges
- Year-month heatmap matrix for any weather metric
- Select aggregation type: Min, Max, or Average
- Color scale options: per year (vertical) or per month (horizontal)
- Compare two locations side by side

### 📈 Tab 2: Seasonal Comparison
- Frequency distribution histograms by season and year
- Displays mean, median, and standard deviations (1σ, 2σ, 3σ)
- Compare mode: different location for the same period, or different period for the same location
- Summary statistics displayed below each chart

### 🔗 Tab 3: Seasonal Correlations
- Correlation matrix heatmap across selected weather metrics
- Automatic percentile-based bucketing (Low: <33%, Med: 33–67%, High: >67%)
- Green = positive correlation, Purple = negative correlation
- Select 2–4 metrics to compare

### 🧭 Tab 4: Wind Rose Correlation
- 8-point compass rose (N, NE, E, SE, S, SW, W, NW)
- Each direction covers a 45-degree sector
- Shows correlation between wind direction and a selected weather metric
- Filled wedge sectors colour-coded by correlation strength

## Data

**Locations:** Kiruna, Luleå, Hemavan, Östersund, Sundsvall, Stockholm, Örebro, Norrköping, Göteborg, Malmö

**Time Period:** 2005-01-01 to 2025-12-31 (20 years, ~76,700 daily observations)

**Weather Variables:**
- Temperature: max, min, mean
- Apparent temperature: max, min, mean
- Daylight duration (hours)
- Sunshine duration (hours)
- Rain sum (mm)
- Snowfall sum (cm)
- Wind speed max (km/h)
- Wind gusts max (km/h)
- Wind direction dominant (degrees)

**Data Source:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Sweden_Historical_Weather_Dashboard.git
cd Sweden_Historical_Weather_Dashboard
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Fetch the Weather Data

```bash
python fetch_weather_data.py
```

This will:
- Download weather data from the Open-Meteo API for all 10 locations
- Save the data to `data/swedish_weather_data.csv`
- Take approximately 10–15 minutes (rate limits require a 65-second pause between locations)
- Use local caching so interrupted runs can be resumed

### 5. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

## File Structure

```
project/
├── app.py                        # Streamlit dashboard application
├── fetch_weather_data.py         # Script to download weather data
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/                         # Created automatically on first fetch
│   └── swedish_weather_data.csv
└── .cache/                       # API cache (created automatically)
```

## Requirements

See `requirements.txt` for the full list. Key dependencies:

- `streamlit` — dashboard framework
- `plotly` — interactive charts
- `pandas` — data handling
- `openmeteo-requests` — Open-Meteo API client
- `requests-cache` — API response caching
- `retry-requests` — automatic retry on API failures
- `scipy` — statistical calculations

## Notes

- The data file is approximately 15–20 MB
- The fetch script uses caching to avoid re-downloading data unnecessarily
- If the fetch is interrupted due to API rate limits, simply re-run the script — cached data will not be re-downloaded
- All 10 locations must be fetched before all map markers appear
