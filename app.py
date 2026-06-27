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
    "wind_gusts_max": "Wind Gusts Max (km/h)",
    "wind_direction_dominant": "Wind Direction (degrees)"
}

# Metrics available for correlation analysis (excluding wind direction)
CORRELATION_METRICS = {k: v for k, v in METRIC_LABELS.items() if k != "wind_direction_dominant"}

SEASON_MONTHS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11]
}

# 8-point compass directions (each covering 45 degrees)
COMPASS_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def get_compass_direction_bin(degree):
    """Convert degree (0-360) to 8-point compass direction based on 45-degree sectors."""
    degree = degree % 360
    if degree >= 0 and degree <= 22.5:
        return "N"
    elif degree > 22.5 and degree <= 67.5:
        return "NE"
    elif degree > 67.5 and degree <= 112.5:
        return "E"
    elif degree > 112.5 and degree <= 157.5:
        return "SE"
    elif degree > 157.5 and degree <= 202.5:
        return "S"
    elif degree > 202.5 and degree <= 247.5:
        return "SW"
    elif degree > 247.5 and degree <= 292.5:
        return "W"
    elif degree > 292.5 and degree <= 337.5:
        return "NW"
    else:
        return "N"


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

    if available_locations:
        display_locations = {k: v for k, v in LOCATIONS.items() if k in available_locations}
    else:
        display_locations = LOCATIONS

    locations_df = pd.DataFrame([
        {"location": name, "lat": coords["lat"], "lon": coords["lon"]}
        for name, coords in display_locations.items()
    ])

    locations_df['selected'] = locations_df['location'] == selected_location

    fig = go.Figure()

    unselected = locations_df[~locations_df['selected']]
    if len(unselected) > 0:
        fig.add_trace(go.Scattergeo(
            lon=unselected['lon'],
            lat=unselected['lat'],
            text=unselected['location'],
            customdata=unselected['location'],
            mode='markers+text',
            marker=dict(size=12, color='#3B82F6', line=dict(width=2, color='white')),
            textposition="top center",
            textfont=dict(size=10, color='#1E293B'),
            name='Locations',
            hovertemplate='<b>%{text}</b><br>Click to select<extra></extra>'
        ))

    selected = locations_df[locations_df['selected']]
    if len(selected) > 0:
        fig.add_trace(go.Scattergeo(
            lon=selected['lon'],
            lat=selected['lat'],
            text=selected['location'],
            customdata=selected['location'],
            mode='markers+text',
            marker=dict(size=20, color='#DC2626', line=dict(width=3, color='white')),
            textposition="top center",
            textfont=dict(size=12, color='#DC2626', family='Arial Black'),
            name='Selected',
            hovertemplate='<b>%{text}</b><br>Selected<extra></extra>'
        ))

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
        geo=dict(projection_scale=3.5, center=dict(lat=63, lon=16)),
        clickmode='event+select'
    )

    return fig


def create_heatmap_matrix(df, location, metric, aggregation, color_scale_direction="vertical"):
    """Create a year-month heatmap matrix for a specific metric."""

    location_data = df[df['location'] == location].copy()

    if aggregation == "Min":
        pivot_data = location_data.groupby(['year', 'month'])[metric].min().reset_index()
    elif aggregation == "Max":
        pivot_data = location_data.groupby(['year', 'month'])[metric].max().reset_index()
    else:
        pivot_data = location_data.groupby(['year', 'month'])[metric].mean().reset_index()

    heatmap_data = pivot_data.pivot(index='month', columns='year', values=metric)
    heatmap_data = heatmap_data.reindex(range(1, 13), fill_value=np.nan)
    heatmap_data = heatmap_data.sort_index(axis=1)

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if color_scale_direction == "horizontal":
        z_values = heatmap_data.values.copy()
        text_values = np.round(heatmap_data.values, 1)
        z_normalized = np.zeros_like(z_values, dtype=float)
        for i in range(z_values.shape[0]):
            row_min = np.nanmin(z_values[i, :])
            row_max = np.nanmax(z_values[i, :])
            if row_max > row_min:
                z_normalized[i, :] = (z_values[i, :] - row_min) / (row_max - row_min)
            else:
                z_normalized[i, :] = 0.5
        colorscale_data = z_normalized
    else:
        colorscale_data = heatmap_data.values
        text_values = np.round(heatmap_data.values, 1)

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
        showscale=(color_scale_direction == "vertical")
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

    location_data = df[df['location'] == location].copy()

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
        return None, None

    values = seasonal_data[metric].dropna()

    if len(values) == 0:
        return None, None

    mean_val = values.mean()
    median_val = values.median()
    std_val = values.std()

    stats_dict = {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'std1_lower': mean_val - std_val,
        'std1_upper': mean_val + std_val,
        'std2_lower': mean_val - 2 * std_val,
        'std2_upper': mean_val + 2 * std_val,
        'std3_lower': mean_val - 3 * std_val,
        'std3_upper': mean_val + 3 * std_val,
    }

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values,
        name='Frequency',
        marker_color='#3B82F6',
        opacity=0.7,
        nbinsx=30
    ))

    # Mean always at top left
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_val:.2f}",
                  annotation_position="top left")

    # Median always at bottom left to avoid overlap
    fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_val:.2f}",
                  annotation_position="bottom left")

    fig.add_vrect(x0=stats_dict['std1_lower'], x1=stats_dict['std1_upper'],
                  fillcolor="yellow", opacity=0.1, layer="below", line_width=0,
                  annotation_text="1σ", annotation_position="top left")

    fig.add_vrect(x0=stats_dict['std2_lower'], x1=stats_dict['std2_upper'],
                  fillcolor="orange", opacity=0.1, layer="below", line_width=0,
                  annotation_text="2σ", annotation_position="top left")

    fig.add_vrect(x0=stats_dict['std3_lower'], x1=stats_dict['std3_upper'],
                  fillcolor="red", opacity=0.05, layer="below", line_width=0,
                  annotation_text="3σ", annotation_position="top left")

    season_label = f"{season} {year}" if season != "Winter" else f"Winter {year - 1}/{year}"

    fig.update_layout(
        title=f"{location} - {METRIC_LABELS[metric]} Distribution ({season_label})",
        xaxis_title=METRIC_LABELS[metric],
        yaxis_title="Frequency",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )

    return fig, stats_dict


def create_wind_rose_correlation(df, location, season, year, metric, bucket_level):
    """Create a wind rose correlation chart."""

    location_data = df[df['location'] == location].copy()

    if season == "Winter":
        seasonal_data = location_data[
            ((location_data['year'] == year - 1) & (location_data['month'] == 12)) |
            ((location_data['year'] == year) & (location_data['month'].isin([1, 2])))
        ].copy()
    else:
        seasonal_data = location_data[
            (location_data['year'] == year) &
            (location_data['month'].isin(SEASON_MONTHS[season]))
        ].copy()

    if len(seasonal_data) == 0:
        return None

    values = seasonal_data[metric].dropna()
    if len(values) == 0:
        return None

    p33 = values.quantile(0.33)
    p67 = values.quantile(0.67)

    # Use .loc to avoid SettingWithCopyWarning
    seasonal_data.loc[:, 'metric_bucket'] = pd.cut(
        seasonal_data[metric],
        bins=[-np.inf, p33, p67, np.inf],
        labels=["Low", "Med", "High"]
    )

    seasonal_data.loc[:, 'compass_direction'] = seasonal_data['wind_direction_dominant'].apply(
        get_compass_direction_bin
    )

    correlations = []

    for direction in COMPASS_DIRECTIONS:
        seasonal_data.loc[:, 'is_direction'] = (seasonal_data['compass_direction'] == direction).astype(int)
        seasonal_data.loc[:, 'is_bucket'] = (seasonal_data['metric_bucket'] == bucket_level).astype(int)

        if seasonal_data['is_direction'].sum() > 0:
            corr = seasonal_data[['is_direction', 'is_bucket']].corr().iloc[0, 1]
            if pd.isna(corr):
                corr = 0
        else:
            corr = 0

        correlations.append(corr)

    radial_positions = [(corr + 1) / 2 * 0.5 + 0.5 for corr in correlations]
    angles = [i * 45 for i in range(8)]

    fig = go.Figure()

    for i in range(len(COMPASS_DIRECTIONS)):
        direction = COMPASS_DIRECTIONS[i]
        angle_deg = angles[i]
        corr = correlations[i]
        radius = radial_positions[i]

        theta_start = angle_deg - 22.5
        theta_end = angle_deg + 22.5
        theta_range = np.linspace(theta_start, theta_end, 20)
        r_range = [radius] * len(theta_range)

        if corr >= 0:
            color_val = corr
            color = f'rgba({int(22 * (1 - color_val) + 22 * color_val)}, {int(163 * color_val)}, {int(52 * color_val)}, 0.7)'
        else:
            color_val = abs(corr)
            color = f'rgba({int(147 * color_val)}, {int(51 * color_val)}, {int(234 * color_val)}, 0.7)'

        fig.add_trace(go.Scatterpolar(
            r=[0.5] + list(r_range) + [0.5],
            theta=[theta_start] + list(theta_range) + [theta_end],
            fill='toself',
            fillcolor=color,
            line=dict(color='white', width=2),
            mode='lines',
            showlegend=False,
            hovertemplate=f'{direction}<br>Correlation: {corr:.3f}<extra></extra>',
            name=direction
        ))

    fig.add_trace(go.Scatterpolar(
        r=radial_positions,
        theta=angles,
        mode='markers+text',
        marker=dict(
            size=12,
            color=correlations,
            colorscale=[[0, '#9333EA'], [0.5, 'white'], [1, '#16A34A']],
            cmin=-1,
            cmax=1,
            colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
            ),
            line=dict(color='white', width=2)
        ),
        text=[f'{c:.2f}' for c in correlations],
        textposition='middle center',
        textfont=dict(size=10, color='black', family='Arial Black'),
        hovertemplate='%{theta}°<br>Correlation: %{r:.3f}<extra></extra>',
        name='Correlation Values',
        showlegend=False
    ))

    zero_circle_r = [0.75] * 100
    zero_circle_theta = list(np.linspace(0, 360, 100))

    fig.add_trace(go.Scatterpolar(
        r=zero_circle_r,
        theta=zero_circle_theta,
        mode='lines',
        line=dict(color='gray', width=2, dash='dot'),
        showlegend=True,
        name='Zero Correlation',
        hoverinfo='skip'
    ))

    season_label = f"{season} {year}" if season != "Winter" else f"Winter {year - 1}/{year}"

    fig.update_layout(
        title=f"{location} - Wind Direction vs {METRIC_LABELS[metric]} ({bucket_level})<br>{season_label}",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.5, 1.0],
                tickvals=[0.5, 0.75, 1.0],
                ticktext=['-1.0<br>(Hub)', '0.0<br>(Middle)', '+1.0<br>(Edge)'],
                showline=True,
                linewidth=2,
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=angles,
                ticktext=COMPASS_DIRECTIONS,
                direction='clockwise',
                rotation=90
            ),
            bgcolor='rgba(240, 240, 240, 0.3)'
        ),
        showlegend=True,
        height=700,
        font=dict(size=12)
    )

    return fig


def create_correlation_matrix(df, location, season, year, metrics_buckets):
    """Create a correlation matrix for selected metrics and their buckets."""

    location_data = df[df['location'] == location].copy()

    if season == "Winter":
        seasonal_data = location_data[
            ((location_data['year'] == year - 1) & (location_data['month'] == 12)) |
            ((location_data['year'] == year) & (location_data['month'].isin([1, 2])))
        ].copy()
    else:
        seasonal_data = location_data[
            (location_data['year'] == year) &
            (location_data['month'].isin(SEASON_MONTHS[season]))
        ].copy()

    if len(seasonal_data) == 0:
        return None

    bucket_columns = []

    for metric, bucket_type in metrics_buckets:
        values = seasonal_data[metric].dropna()

        if len(values) == 0:
            continue

        p33 = values.quantile(0.33)
        p67 = values.quantile(0.67)

        bucket_col = f"{metric}_bucket"
        seasonal_data.loc[:, bucket_col] = pd.cut(
            seasonal_data[metric],
            bins=[-np.inf, p33, p67, np.inf],
            labels=[f"{METRIC_LABELS[metric]} Low",
                    f"{METRIC_LABELS[metric]} Med",
                    f"{METRIC_LABELS[metric]} High"]
        )

        bucket_columns.append(bucket_col)

    bucket_dummies = pd.get_dummies(seasonal_data[bucket_columns])
    corr_matrix = bucket_dummies.corr()

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

    season_label = f"{season} {year}" if season != "Winter" else f"Winter {year - 1}/{year}"

    fig.update_layout(
        title=f"{location} - Weather Metric Correlations ({season_label})",
        xaxis_title="",
        yaxis_title="",
        height=600,
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickangle=0)
    )

    return fig


def handle_map_click(selected_points):
    """Handle map click events and update selected location."""
    if selected_points and selected_points.selection and 'points' in selected_points.selection:
        points = selected_points.selection['points']
        if len(points) > 0 and 'customdata' in points[0]:
            clicked_location = points[0]['customdata']
            if clicked_location != st.session_state.selected_location:
                st.session_state.selected_location = clicked_location
                st.rerun()


def render_map_column(available_locations, tab_key):
    """Render the map column with location selector - shared across all tabs."""
    st.subheader("Select Location")

    if st.session_state.selected_location is None:
        st.info("👆 Click on a location on the map to begin")
    else:
        st.success(f"Selected: **{st.session_state.selected_location}**")

    map_fig = create_sweden_map(st.session_state.selected_location, available_locations)
    selected_points = st.plotly_chart(map_fig, width='stretch', key=f"map_{tab_key}", on_select="rerun")
    handle_map_click(selected_points)

    st.markdown("---")
    st.markdown("**Or select from list:**")
    current_index = (
        0 if st.session_state.selected_location is None
        else available_locations.index(st.session_state.selected_location) + 1
    )
    selected = st.selectbox(
        "Location",
        options=[None] + available_locations,
        index=current_index,
        key=f"location_select_{tab_key}"
    )

    if selected and selected != st.session_state.selected_location:
        st.session_state.selected_location = selected
        st.rerun()


def main():
    """Main application function."""

    df = load_data()
    available_locations = sorted(df['location'].unique())

    if 'selected_location' not in st.session_state:
        st.session_state.selected_location = None

    st.markdown('<div class="main-title">🇸🇪 Swedish Weather Data Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">20 Years of Historical Weather Data (2005-2025)</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Metric Ranges",
        "📈 Seasonal Comparison",
        "🔗 Seasonal Correlations",
        "🧭 Wind Rose Correlation"
    ])

    # -------------------------
    # TAB 1: Metric Ranges
    # -------------------------
    with tab1:
        col_map, col_viz = st.columns([1, 2])

        with col_map:
            render_map_column(available_locations, "tab1")

        with col_viz:
            if st.session_state.selected_location:
                st.subheader("Metric Heatmap")

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

                heatmap_fig = create_heatmap_matrix(
                    df, st.session_state.selected_location, metric, aggregation, color_direction
                )
                st.plotly_chart(heatmap_fig, width='stretch')

                if compare:
                    other_locations = [loc for loc in available_locations if loc != st.session_state.selected_location]
                    compare_location = st.selectbox(
                        "Compare with:",
                        options=other_locations,
                        key="compare_location_tab1"
                    )

                    if compare_location:
                        heatmap_fig2 = create_heatmap_matrix(
                            df, compare_location, metric, aggregation, color_direction
                        )
                        st.plotly_chart(heatmap_fig2, width='stretch')
            else:
                st.info("👈 Please select a location from the map to view data")

    # -------------------------
    # TAB 2: Seasonal Comparison
    # -------------------------
    with tab2:
        col_map, col_viz = st.columns([1, 2])

        with col_map:
            render_map_column(available_locations, "tab2")

        with col_viz:
            if st.session_state.selected_location:
                st.subheader("Seasonal Distribution")

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

                dist_fig, stats = create_seasonal_distribution(
                    df, st.session_state.selected_location, metric, season, year
                )

                if dist_fig:
                    st.plotly_chart(dist_fig, width='stretch')

                    if stats:
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        col_s1.metric("Mean", f"{stats['mean']:.2f}")
                        col_s2.metric("Median", f"{stats['median']:.2f}")
                        col_s3.metric("Std Dev", f"{stats['std']:.2f}")
                        col_s4.metric("Range", f"{stats['std3_upper'] - stats['std3_lower']:.2f}")
                else:
                    st.warning("No data available for selected period")

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
                            st.plotly_chart(dist_fig2, width='stretch')

                            if stats2:
                                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                                col_s1.metric("Mean", f"{stats2['mean']:.2f}")
                                col_s2.metric("Median", f"{stats2['median']:.2f}")
                                col_s3.metric("Std Dev", f"{stats2['std']:.2f}")
                                col_s4.metric("Range", f"{stats2['std3_upper'] - stats2['std3_lower']:.2f}")

                    else:
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
                            st.plotly_chart(dist_fig2, width='stretch')

                            if stats2:
                                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                                col_s1.metric("Mean", f"{stats2['mean']:.2f}")
                                col_s2.metric("Median", f"{stats2['median']:.2f}")
                                col_s3.metric("Std Dev", f"{stats2['std']:.2f}")
                                col_s4.metric("Range", f"{stats2['std3_upper'] - stats2['std3_lower']:.2f}")
            else:
                st.info("👈 Please select a location from the map to view data")

    # -------------------------
    # TAB 3: Seasonal Correlations
    # -------------------------
    with tab3:
        col_map, col_viz = st.columns([1, 2])

        with col_map:
            render_map_column(available_locations, "tab3")

        with col_viz:
            if st.session_state.selected_location:
                st.subheader("Correlation Matrix")

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

                num_metrics = st.slider("Number of metrics to compare:", 2, 4, 2, key="num_metrics_tab3")

                metrics_buckets = []
                for i in range(num_metrics):
                    metric = st.selectbox(
                        f"Metric {i + 1}",
                        options=list(METRIC_LABELS.keys()),
                        format_func=lambda x: METRIC_LABELS[x],
                        key=f"metric_{i}_tab3"
                    )
                    metrics_buckets.append((metric, "percentile"))

                if st.button("Generate Correlation Matrix", key="generate_corr"):
                    corr_fig = create_correlation_matrix(
                        df, st.session_state.selected_location, season, year, metrics_buckets
                    )

                    if corr_fig:
                        st.plotly_chart(corr_fig, width='stretch')

                        st.markdown("---")
                        st.markdown("**Interpretation:**")
                        st.markdown("🟢 **Green**: Strong positive correlation")
                        st.markdown("⚪ **White**: No correlation")
                        st.markdown("🟣 **Purple**: Strong negative correlation")
                    else:
                        st.warning("Insufficient data for selected period")
            else:
                st.info("👈 Please select a location from the map to view data")

    # -------------------------
    # TAB 4: Wind Rose Correlation
    # -------------------------
    with tab4:
        col_map, col_viz = st.columns([1, 2])

        with col_map:
            render_map_column(available_locations, "tab4")

        with col_viz:
            if st.session_state.selected_location:
                st.subheader("Wind Rose Correlation")

                st.markdown("""
                This chart shows the correlation between wind direction and a selected weather metric.
                - **Green (outer)**: Positive correlation - this wind direction is associated with higher values
                - **Purple (inner)**: Negative correlation - this wind direction is associated with lower values
                - **Dotted circle**: Zero correlation baseline
                """)

                col1, col2 = st.columns(2)

                with col1:
                    season = st.selectbox(
                        "Season",
                        options=list(SEASON_MONTHS.keys()),
                        key="season_tab4"
                    )

                with col2:
                    if season == "Winter":
                        year = st.selectbox(
                            "Year (ending)",
                            options=range(df['year'].min() + 1, df['year'].max() + 1),
                            index=0,
                            key="year_tab4"
                        )
                    else:
                        year = st.selectbox(
                            "Year",
                            options=range(df['year'].min(), df['year'].max() + 1),
                            index=0,
                            key="year_tab4_regular"
                        )

                col3, col4 = st.columns(2)

                with col3:
                    metric = st.selectbox(
                        "Weather Metric",
                        options=list(CORRELATION_METRICS.keys()),
                        format_func=lambda x: CORRELATION_METRICS[x],
                        key="metric_tab4"
                    )

                with col4:
                    bucket = st.selectbox(
                        "Metric Level",
                        options=["Low", "Med", "High"],
                        index=2,
                        key="bucket_tab4",
                        help="Low: bottom 33%, Med: middle 33%, High: top 33%"
                    )

                if st.button("Generate Wind Rose", key="generate_windrose"):
                    with st.spinner("Calculating correlations..."):
                        windrose_fig = create_wind_rose_correlation(
                            df, st.session_state.selected_location, season, year, metric, bucket
                        )

                        if windrose_fig:
                            st.plotly_chart(windrose_fig, width='stretch')

                            st.markdown("---")
                            st.markdown("**How to interpret:**")
                            st.markdown(f"- Points closer to the **edge** indicate wind directions strongly associated with **{bucket.lower()} {CORRELATION_METRICS[metric].lower()}**")
                            st.markdown(f"- Points closer to the **hub** indicate wind directions associated with the **opposite** of {bucket.lower()} values")
                            st.markdown("- Points near the **dotted circle** show little to no association")
                        else:
                            st.warning("Insufficient data for selected period")
            else:
                st.info("👈 Please select a location from the map to view data")


if __name__ == "__main__":
    main()
