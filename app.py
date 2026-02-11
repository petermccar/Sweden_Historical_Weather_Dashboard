"""
Swedish Weather Data Visualization Dashboard
Three-tab interactive dashboard with map-based location selection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import os

# Page configuration
st.set_page_config(
    page_title="Swedish Weather Dashboard",
    page_icon="🇸🇪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Location coordinates
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

# Weather metric mappings
METRIC_LABELS = {
    "temperature_max": "Temperature Max (°C)",
    "temperature_min": "Temperature Min (°C)",
    "temperature_mean": "Temperature Mean (°C)",
    "apparent_temperature_max": "Apparent Temp Max (°C)",
    "apparent_temperature_min": "Apparent Temp Min (°C)",
    "apparent_temperature_mean": "Apparent Temp Mean (°C)",
    "daylight_duration": "Daylight Duration (hours)",
    "sunshine_duration": "Sunshine Duration (hours)",
    "rain_sum": "Rain Sum (mm)",
    "snowfall_sum": "Snowfall Sum (cm)",
    "wind_speed_max": "Wind Speed Max (km/h)",
    "wind_gusts_max": "Wind Gusts Max (km/h)"
}

SEASON_MONTHS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11]
}


@st.cache_data
def load_data():
    """Load weather data from CSV file."""
    data_path = "data/swedish_weather_data.csv"
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}. Please run fetch_weather_data.py first.")
        st.stop()
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].apply(get_season)
    
    # Convert daylight and sunshine duration from seconds to hours
    df['daylight_duration'] = df['daylight_duration'] / 3600
    df['sunshine_duration'] = df['sunshine_duration'] / 3600
    
    return df


def get_season(month):
    """Get season name from month number."""
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    return "Unknown"


def create_sweden_map(selected_location=None, available_locations=None):
    """Create an interactive map of Sweden with location markers."""
    
    # Filter locations based on available data
    if available_locations:
        display_locations = {k: v for k, v in LOCATIONS.items() if k in available_locations}
    else:
        display_locations = LOCATIONS
    
    # Create location dataframe
    locations_df = pd.DataFrame([
        {"location": name, "lat": coords["lat"], "lon": coords["lon"]}
        for name, coords in display_locations.items()
    ])
    
    # Add selection indicator
    locations_df['selected'] = locations_df['location'] == selected_location
    locations_df['size'] = locations_df['selected'].apply(lambda x: 20 if x else 12)
    locations_df['color'] = locations_df['selected'].apply(lambda x: '#DC2626' if x else '#3B82F6')
    
    # Create map centered on Sweden
    fig = go.Figure()
    
    # Add unselected locations
    unselected = locations_df[~locations_df['selected']]
    if len(unselected) > 0:
        fig.add_trace(go.Scattergeo(
            lon=unselected['lon'],
            lat=unselected['lat'],
            text=unselected['location'],
            customdata=unselected['location'],  # Add customdata for click handling
            mode='markers+text',
            marker=dict(size=12, color='#3B82F6', line=dict(width=2, color='white')),
            textposition="top center",
            textfont=dict(size=10, color='#1E293B'),
            name='Locations',
            hovertemplate='<b>%{text}</b><br>Click to select<extra></extra>'
        ))
    
    # Add selected location
    selected = locations_df[locations_df['selected']]
    if len(selected) > 0:
        fig.add_trace(go.Scattergeo(
            lon=selected['lon'],
            lat=selected['lat'],
            text=selected['location'],
            customdata=selected['location'],  # Add customdata for click handling
            mode='markers+text',
            marker=dict(size=20, color='#DC2626', line=dict(width=3, color='white')),
            textposition="top center",
            textfont=dict(size=12, color='#DC2626', family='Arial Black'),
            name='Selected',
            hovertemplate='<b>%{text}</b><br>Selected<extra></extra>'
        ))
    
    # Map styling
    fig.update_geos(
        scope='europe',
        center=dict(lat=63, lon=16),
        projection_scale=3.5,
        showland=True,
        landcolor='#F1F5F9',
        showlakes=True,
        lakecolor='#BAE6FD',
        showcountries=True,
        countrycolor='#CBD5E1',
        showcoastlines=True,
        coastlinecolor='#64748B',
        projection_type='mercator',
        bgcolor='#E0F2FE'
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        geo=dict(
            projection_scale=3.5,
            center=dict(lat=63, lon=16)
        ),
        clickmode='event+select'  # Enable click events
    )
    
    return fig


def create_heatmap_matrix(df, location, metric, aggregation, color_scale_direction="vertical"):
    """Create a year-month heatmap matrix for a specific metric.
    
    Args:
        df: DataFrame with weather data
        location: Location name
        metric: Weather metric to display
        aggregation: "Min", "Max", or "Average"
        color_scale_direction: "vertical" (per year) or "horizontal" (per month)
    """
    
    # Filter data for location
    location_data = df[df['location'] == location].copy()
    
    # Create year-month aggregation
    if aggregation == "Min":
        pivot_data = location_data.groupby(['year', 'month'])[metric].min().reset_index()
    elif aggregation == "Max":
        pivot_data = location_data.groupby(['year', 'month'])[metric].max().reset_index()
    else:  # Average
        pivot_data = location_data.groupby(['year', 'month'])[metric].mean().reset_index()
    
    # Create pivot table
    heatmap_data = pivot_data.pivot(index='month', columns='year', values=metric)
    
    # Ensure all months are present
    heatmap_data = heatmap_data.reindex(range(1, 13), fill_value=np.nan)
    
    # Sort years
    heatmap_data = heatmap_data.sort_index(axis=1)
    
    # Month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Determine color scaling based on direction
    if color_scale_direction == "horizontal":
        # Scale colors per month (across all years in that row)
        z_values = heatmap_data.values.copy()
        text_values = np.round(heatmap_data.values, 1)
        
        # Normalize each row separately for color scaling
        z_normalized = np.zeros_like(z_values, dtype=float)
        for i in range(z_values.shape[0]):
            row_min = np.nanmin(z_values[i, :])
            row_max = np.nanmax(z_values[i, :])
            if row_max > row_min:
                z_normalized[i, :] = (z_values[i, :] - row_min) / (row_max - row_min)
            else:
                z_normalized[i, :] = 0.5  # If all values same, use middle color
        
        # Use normalized values for colors
        colorscale_data = z_normalized
    else:
        # Default: Scale colors per year (down each column) - this is the original behavior
        colorscale_data = heatmap_data.values
        text_values = np.round(heatmap_data.values, 1)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=colorscale_data,
        x=heatmap_data.columns,
        y=month_labels,
        colorscale='RdBu_r',
        text=text_values,
        texttemplate='%{text}',
        textfont={"size": 10, "color": "black"},
        hovertemplate='Year: %{x}<br>Month: %{y}<br>Value: %{text}<extra></extra>',
        colorbar=dict(title=METRIC_LABELS[metric]),
        showscale=(color_scale_direction == "vertical")  # Only show colorbar for vertical
    ))
    
    scale_note = "per year" if color_scale_direction == "vertical" else "per month"
    
    fig.update_layout(
        title=f"{location} - {METRIC_LABELS[metric]} ({aggregation}) - Color scale: {scale_note}",
        xaxis_title="Year",
        yaxis_title="Month",
        height=400,
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickmode='array', tickvals=list(range(12)), ticktext=month_labels)
    )
    
    return fig


def create_seasonal_distribution(df, location, metric, season, year):
    """Create a frequency distribution histogram for seasonal data."""
    
    # Filter data
    location_data = df[df['location'] == location].copy()
    
    # Handle winter season (spans two calendar years)
    if season == "Winter":
        # Winter includes Dec of previous year and Jan-Feb of current year
        seasonal_data = location_data[
            ((location_data['year'] == year - 1) & (location_data['month'] == 12)) |
            ((location_data['year'] == year) & (location_data['month'].isin([1, 2])))
        ]
    else:
        seasonal_data = location_data[
            (location_data['year'] == year) & 
            (location_data['month'].isin(SEASON_MONTHS[season]))
        ]
    
    if len(seasonal_data) == 0:
        return None, None
    
    values = seasonal_data[metric].dropna()
    
    if len(values) == 0:
        return None, None
    
    # Calculate statistics
    mean_val = values.mean()
    median_val = values.median()
    std_val = values.std()
    
    stats_dict = {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'std1_lower': mean_val - std_val,
        'std1_upper': mean_val + std_val,
        'std2_lower': mean_val - 2*std_val,
        'std2_upper': mean_val + 2*std_val,
        'std3_lower': mean_val - 3*std_val,
        'std3_upper': mean_val + 3*std_val,
    }
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=values,
        name='Frequency',
        marker_color='#3B82F6',
        opacity=0.7,
        nbinsx=30
    ))
    
    # Add mean line
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_val:.2f}", 
                  annotation_position="top left")
    
    # Add median line - always position at bottom to avoid overlap
    fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_val:.2f}",
                  annotation_position="bottom left")
    
    # Add standard deviation ranges
    fig.add_vrect(x0=stats_dict['std1_lower'], x1=stats_dict['std1_upper'],
                  fillcolor="yellow", opacity=0.1, layer="below", line_width=0,
                  annotation_text="1σ", annotation_position="top left")
    
    fig.add_vrect(x0=stats_dict['std2_lower'], x1=stats_dict['std2_upper'],
                  fillcolor="orange", opacity=0.1, layer="below", line_width=0,
                  annotation_text="2σ", annotation_position="top left")
    
    fig.add_vrect(x0=stats_dict['std3_lower'], x1=stats_dict['std3_upper'],
                  fillcolor="red", opacity=0.05, layer="below", line_width=0,
                  annotation_text="3σ", annotation_position="top left")
    
    season_label = f"{season} {year}" if season != "Winter" else f"Winter {year-1}/{year}"
    
    fig.update_layout(
        title=f"{location} - {METRIC_LABELS[metric]} Distribution ({season_label})",
        xaxis_title=METRIC_LABELS[metric],
        yaxis_title="Frequency",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig, stats_dict


def create_correlation_matrix(df, location, season, year, metrics_buckets):
    """Create a correlation matrix for selected metrics and their buckets."""
    
    # Filter data for season and location
    location_data = df[df['location'] == location].copy()
    
    # Handle winter season
    if season == "Winter":
        seasonal_data = location_data[
            ((location_data['year'] == year - 1) & (location_data['month'] == 12)) |
            ((location_data['year'] == year) & (location_data['month'].isin([1, 2])))
        ]
    else:
        seasonal_data = location_data[
            (location_data['year'] == year) & 
            (location_data['month'].isin(SEASON_MONTHS[season]))
        ]
    
    if len(seasonal_data) == 0:
        return None
    
    # Create buckets for each metric
    bucket_columns = []
    bucket_labels = []
    
    for metric, bucket_type in metrics_buckets:
        values = seasonal_data[metric].dropna()
        
        if len(values) == 0:
            continue
        
        # Calculate percentile-based buckets
        p33 = values.quantile(0.33)
        p67 = values.quantile(0.67)
        
        # Create bucket column
        bucket_col = f"{metric}_bucket"
        seasonal_data[bucket_col] = pd.cut(
            seasonal_data[metric],
            bins=[-np.inf, p33, p67, np.inf],
            labels=[f"{METRIC_LABELS[metric]} Low", 
                   f"{METRIC_LABELS[metric]} Med",
                   f"{METRIC_LABELS[metric]} High"]
        )
        
        bucket_columns.append(bucket_col)
        bucket_labels.extend([f"{METRIC_LABELS[metric]} Low", 
                             f"{METRIC_LABELS[metric]} Med",
                             f"{METRIC_LABELS[metric]} High"])
    
    # Create dummy variables for all buckets
    bucket_dummies = pd.get_dummies(seasonal_data[bucket_columns])
    
    # Calculate correlation matrix
    corr_matrix = bucket_dummies.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[[0, '#9333EA'], [0.5, 'white'], [1, '#16A34A']],
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 9},
        hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(title="Correlation")
    ))
    
    season_label = f"{season} {year}" if season != "Winter" else f"Winter {year-1}/{year}"
    
    fig.update_layout(
        title=f"{location} - Weather Metric Correlations ({season_label})",
        xaxis_title="",
        yaxis_title="",
        height=600,
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickangle=0)
    )
    
    return fig


def main():
    """Main application function."""
    
    # Load data
    df = load_data()
    
    # Get available locations from data
    available_locations = sorted(df['location'].unique())
    
    # Initialize session state
    if 'selected_location' not in st.session_state:
        st.session_state.selected_location = None
    if 'compare_mode' not in st.session_state:
        st.session_state.compare_mode = False
    if 'compare_location' not in st.session_state:
        st.session_state.compare_location = None
    
    # Header
    st.markdown('<div class="main-title">🇸🇪 Swedish Weather Data Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">20 Years of Historical Weather Data (2005-2025)</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Metric Ranges", "📈 Seasonal Comparison", "🔗 Seasonal Correlations"])
    
    # TAB 1: Metric Ranges
    with tab1:
        col_map, col_viz = st.columns([1, 2])
        
        with col_map:
            st.subheader("Select Location")
            
            if st.session_state.selected_location is None:
                st.info("👆 Click on a location on the map to begin")
            else:
                st.success(f"Selected: **{st.session_state.selected_location}**")
            
            # Display map
            map_fig = create_sweden_map(st.session_state.selected_location, available_locations)
            selected_points = st.plotly_chart(map_fig, use_container_width=True, key="map_tab1", on_select="rerun")
            
            # Handle map clicks
            if selected_points and selected_points.selection and 'points' in selected_points.selection:
                points = selected_points.selection['points']
                if len(points) > 0 and 'customdata' in points[0]:
                    clicked_location = points[0]['customdata']
                    if clicked_location != st.session_state.selected_location:
                        st.session_state.selected_location = clicked_location
                        st.rerun()
            
            # Location selector buttons (as backup to map clicking)
            st.markdown("---")
            st.markdown("**Or select from list:**")
            selected = st.selectbox(
                "Location",
                options=[None] + available_locations,
                index=0 if st.session_state.selected_location is None else available_locations.index(st.session_state.selected_location) + 1,
                key="location_select_tab1"
            )
            
            if selected and selected != st.session_state.selected_location:
                st.session_state.selected_location = selected
                st.rerun()
        
        with col_viz:
            if st.session_state.selected_location:
                st.subheader("Metric Heatmap")
                
                # Controls
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    metric = st.selectbox(
                        "Weather Metric",
                        options=list(METRIC_LABELS.keys()),
                        format_func=lambda x: METRIC_LABELS[x],
                        key="metric_tab1"
                    )
                
                with col2:
                    aggregation = st.selectbox(
                        "Aggregation Type",
                        options=["Min", "Max", "Average"],
                        key="agg_tab1"
                    )
                
                with col3:
                    color_direction = st.selectbox(
                        "Color Scale",
                        options=["vertical", "horizontal"],
                        format_func=lambda x: "Per Year" if x == "vertical" else "Per Month",
                        key="color_direction_tab1"
                    )
                
                with col4:
                    compare = st.checkbox("Compare Locations", key="compare_tab1")
                
                st.info("ℹ️ Metrics based on daily observations")
                
                # Create heatmap for primary location
                heatmap_fig = create_heatmap_matrix(df, st.session_state.selected_location, metric, aggregation, color_direction)
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Compare mode
                if compare:
                    other_locations = [loc for loc in available_locations if loc != st.session_state.selected_location]
                    compare_location = st.selectbox(
                        "Compare with:",
                        options=other_locations,
                        key="compare_location_tab1"
                    )
                    
                    if compare_location:
                        heatmap_fig2 = create_heatmap_matrix(df, compare_location, metric, aggregation, color_direction)
                        st.plotly_chart(heatmap_fig2, use_container_width=True)
            else:
                st.info("👈 Please select a location from the map to view data")
    
    # TAB 2: Seasonal Comparison
    with tab2:
        col_map, col_viz = st.columns([1, 2])
        
        with col_map:
            st.subheader("Select Location")
            
            if st.session_state.selected_location:
                st.success(f"Selected: **{st.session_state.selected_location}**")
            
            # Display map
            map_fig = create_sweden_map(st.session_state.selected_location, available_locations)
            selected_points = st.plotly_chart(map_fig, use_container_width=True, key="map_tab2", on_select="rerun")
            
            # Handle map clicks
            if selected_points and selected_points.selection and 'points' in selected_points.selection:
                points = selected_points.selection['points']
                if len(points) > 0 and 'customdata' in points[0]:
                    clicked_location = points[0]['customdata']
                    if clicked_location != st.session_state.selected_location:
                        st.session_state.selected_location = clicked_location
                        st.rerun()
            
            # Location selector
            st.markdown("---")
            st.markdown("**Or select from list:**")
            selected = st.selectbox(
                "Location",
                options=[None] + available_locations,
                index=0 if st.session_state.selected_location is None else available_locations.index(st.session_state.selected_location) + 1,
                key="location_select_tab2"
            )
            
            if selected and selected != st.session_state.selected_location:
                st.session_state.selected_location = selected
                st.rerun()
        
        with col_viz:
            if st.session_state.selected_location:
                st.subheader("Seasonal Distribution")
                
                # Controls
                col1, col2 = st.columns(2)
                
                with col1:
                    metric = st.selectbox(
                        "Weather Metric",
                        options=list(METRIC_LABELS.keys()),
                        format_func=lambda x: METRIC_LABELS[x],
                        key="metric_tab2"
                    )
                
                with col2:
                    season = st.selectbox(
                        "Season",
                        options=list(SEASON_MONTHS.keys()),
                        key="season_tab2"
                    )
                
                # Year selection (adjust for winter)
                min_year = df['year'].min()
                max_year = df['year'].max()
                
                if season == "Winter":
                    year = st.selectbox(
                        "Year (ending)",
                        options=range(min_year + 1, max_year + 1),
                        index=0,
                        key="year_tab2"
                    )
                else:
                    year = st.selectbox(
                        "Year",
                        options=range(min_year, max_year + 1),
                        index=0,
                        key="year_tab2_regular"
                    )
                
                compare_mode = st.checkbox("Compare Mode", key="compare_mode_tab2")
                
                # Create primary distribution
                dist_fig, stats = create_seasonal_distribution(
                    df, st.session_state.selected_location, metric, season, year
                )
                
                if dist_fig:
                    st.plotly_chart(dist_fig, use_container_width=True)
                    
                    # Display statistics
                    if stats:
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        col_s1.metric("Mean", f"{stats['mean']:.2f}")
                        col_s2.metric("Median", f"{stats['median']:.2f}")
                        col_s3.metric("Std Dev", f"{stats['std']:.2f}")
                        col_s4.metric("Range", f"{stats['std3_upper']-stats['std3_lower']:.2f}")
                else:
                    st.warning("No data available for selected period")
                
                # Compare mode
                if compare_mode:
                    compare_type = st.radio(
                        "Compare by:",
                        options=["Different Location (same period)", "Different Period (same location)"],
                        key="compare_type_tab2"
                    )
                    
                    if compare_type == "Different Location (same period)":
                        other_locations = [loc for loc in available_locations if loc != st.session_state.selected_location]
                        compare_location = st.selectbox(
                            "Compare with location:",
                            options=other_locations,
                            key="compare_location_tab2"
                        )
                        
                        dist_fig2, stats2 = create_seasonal_distribution(
                            df, compare_location, metric, season, year
                        )
                        
                        if dist_fig2:
                            st.plotly_chart(dist_fig2, use_container_width=True)
                            
                            # Display statistics for comparison
                            if stats2:
                                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                                col_s1.metric("Mean", f"{stats2['mean']:.2f}")
                                col_s2.metric("Median", f"{stats2['median']:.2f}")
                                col_s3.metric("Std Dev", f"{stats2['std']:.2f}")
                                col_s4.metric("Range", f"{stats2['std3_upper']-stats2['std3_lower']:.2f}")
                    
                    else:  # Different period
                        col_s1, col_s2 = st.columns(2)
                        
                        with col_s1:
                            season2 = st.selectbox(
                                "Season",
                                options=list(SEASON_MONTHS.keys()),
                                key="season2_tab2"
                            )
                        
                        with col_s2:
                            if season2 == "Winter":
                                year2 = st.selectbox(
                                    "Year (ending)",
                                    options=range(min_year + 1, max_year + 1),
                                    index=1 if max_year > min_year + 1 else 0,
                                    key="year2_tab2"
                                )
                            else:
                                year2 = st.selectbox(
                                    "Year",
                                    options=range(min_year, max_year + 1),
                                    index=1 if max_year > min_year else 0,
                                    key="year2_tab2_regular"
                                )
                        
                        dist_fig2, stats2 = create_seasonal_distribution(
                            df, st.session_state.selected_location, metric, season2, year2
                        )
                        
                        if dist_fig2:
                            st.plotly_chart(dist_fig2, use_container_width=True)
                            
                            # Display statistics for comparison
                            if stats2:
                                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                                col_s1.metric("Mean", f"{stats2['mean']:.2f}")
                                col_s2.metric("Median", f"{stats2['median']:.2f}")
                                col_s3.metric("Std Dev", f"{stats2['std']:.2f}")
                                col_s4.metric("Range", f"{stats2['std3_upper']-stats2['std3_lower']:.2f}")
            else:
                st.info("👈 Please select a location from the map to view data")
    
    # TAB 3: Seasonal Correlations
    with tab3:
        col_map, col_viz = st.columns([1, 2])
        
        with col_map:
            st.subheader("Select Location")
            
            if st.session_state.selected_location:
                st.success(f"Selected: **{st.session_state.selected_location}**")
            
            # Display map
            map_fig = create_sweden_map(st.session_state.selected_location, available_locations)
            selected_points = st.plotly_chart(map_fig, use_container_width=True, key="map_tab3", on_select="rerun")
            
            # Handle map clicks
            if selected_points and selected_points.selection and 'points' in selected_points.selection:
                points = selected_points.selection['points']
                if len(points) > 0 and 'customdata' in points[0]:
                    clicked_location = points[0]['customdata']
                    if clicked_location != st.session_state.selected_location:
                        st.session_state.selected_location = clicked_location
                        st.rerun()
            
            # Location selector
            st.markdown("---")
            st.markdown("**Or select from list:**")
            selected = st.selectbox(
                "Location",
                options=[None] + available_locations,
                index=0 if st.session_state.selected_location is None else available_locations.index(st.session_state.selected_location) + 1,
                key="location_select_tab3"
            )
            
            if selected and selected != st.session_state.selected_location:
                st.session_state.selected_location = selected
                st.rerun()
        
        with col_viz:
            if st.session_state.selected_location:
                st.subheader("Correlation Matrix")
                
                # Controls
                col1, col2 = st.columns(2)
                
                with col1:
                    season = st.selectbox(
                        "Season",
                        options=list(SEASON_MONTHS.keys()),
                        key="season_tab3"
                    )
                
                with col2:
                    if season == "Winter":
                        year = st.selectbox(
                            "Year (ending)",
                            options=range(df['year'].min() + 1, df['year'].max() + 1),
                            index=0,
                            key="year_tab3"
                        )
                    else:
                        year = st.selectbox(
                            "Year",
                            options=range(df['year'].min(), df['year'].max() + 1),
                            index=0,
                            key="year_tab3_regular"
                        )
                
                st.markdown("**Select metrics to correlate:**")
                st.info("Buckets are automatically created using percentiles (Low: <33%, Med: 33-67%, High: >67%)")
                
                # Metric selection
                num_metrics = st.slider("Number of metrics to compare:", 2, 4, 2, key="num_metrics_tab3")
                
                metrics_buckets = []
                for i in range(num_metrics):
                    metric = st.selectbox(
                        f"Metric {i+1}",
                        options=list(METRIC_LABELS.keys()),
                        format_func=lambda x: METRIC_LABELS[x],
                        key=f"metric_{i}_tab3"
                    )
                    metrics_buckets.append((metric, "percentile"))
                
                # Create correlation matrix
                if st.button("Generate Correlation Matrix", key="generate_corr"):
                    corr_fig = create_correlation_matrix(
                        df, st.session_state.selected_location, season, year, metrics_buckets
                    )
                    
                    if corr_fig:
                        st.plotly_chart(corr_fig, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("**Interpretation:**")
                        st.markdown("🟢 **Green**: Strong positive correlation")
                        st.markdown("⚪ **White**: No correlation")
                        st.markdown("🟣 **Purple**: Strong negative correlation")
                    else:
                        st.warning("Insufficient data for selected period")
            else:
                st.info("👈 Please select a location from the map to view data")


if __name__ == "__main__":
    main()
