"""
Weather Analyst — AgroSage
Historical climate analysis and mock weather fallback for agricultural planning.
"""

import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import load_css, render_sidebar
from utils.weather_utils import load_climate_csv, get_monthly_stats, get_mock_weather

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Weather Analyst — AgroSage",
    page_icon="Ag",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
render_sidebar()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLIMATE_CSV = os.path.join(DATA_DIR, "DailyDelhiClimate.csv")

# ── Page-specific styles ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Section label ── */
.wa-section {
    margin-top: 1.75rem;
    margin-bottom: 0.5rem;
}
.wa-section-title {
    font-family: var(--font-body);
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--color-border-light);
}

/* ── Mock weather card ── */
.wa-mock-card {
    background: linear-gradient(135deg, #1B6B93 0%, #4FC0D0 50%, #A8E6CF 100%);
    border-radius: var(--radius-lg);
    padding: 1.75rem 2rem;
    color: #FFFFFF;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
}
.wa-mock-card::before {
    content: '';
    position: absolute;
    top: -30px;
    right: -30px;
    width: 120px;
    height: 120px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}
.wa-mock-card::after {
    content: '';
    position: absolute;
    bottom: -20px;
    right: 40px;
    width: 80px;
    height: 80px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.wa-mock-eyebrow {
    font-family: var(--font-body);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.7;
    margin-bottom: 0.25rem;
}
.wa-mock-city {
    font-family: var(--font-body);
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
    margin-bottom: 0.1rem;
}
.wa-mock-climate {
    font-family: var(--font-body);
    font-size: 0.82rem;
    opacity: 0.7;
    margin-bottom: 0.85rem;
}
.wa-mock-temp {
    font-family: var(--font-mono);
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
}
.wa-mock-condition {
    font-family: var(--font-body);
    font-size: 0.92rem;
    font-weight: 500;
    margin-top: 0.3rem;
    opacity: 0.85;
}
.wa-mock-details {
    display: flex;
    gap: 1.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.wa-mock-detail {
    display: flex;
    flex-direction: column;
}
.wa-mock-detail-label {
    font-family: var(--font-body);
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.6;
}
.wa-mock-detail-value {
    font-family: var(--font-mono);
    font-size: 0.92rem;
    font-weight: 600;
    margin-top: 0.1rem;
}

/* ── Forecast stripe ── */
.wa-forecast-row {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    padding: 0.7rem 1rem;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: var(--transition);
}
.wa-forecast-row:hover {
    border-color: var(--color-accent);
}
.wa-forecast-day {
    font-family: var(--font-body);
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--color-primary);
    min-width: 90px;
}
.wa-forecast-cond {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--color-text-secondary);
    flex: 1;
    margin-left: 0.75rem;
}
.wa-forecast-temps {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--color-text);
    font-weight: 500;
}
.wa-forecast-rain {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--color-info);
    min-width: 50px;
    text-align: right;
}

/* ── Compact stat card ── */
.wa-stat {
    background: var(--color-surface);
    border: 1px solid var(--color-border-light);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    transition: var(--transition);
}
.wa-stat:hover {
    border-color: var(--color-accent);
    box-shadow: var(--shadow-sm);
}
.wa-stat-label {
    font-family: var(--font-body);
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.15rem;
}
.wa-stat-value {
    font-family: var(--font-mono);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--color-primary);
}

/* ── Note card ── */
.wa-note {
    background: var(--color-surface-alt);
    border: 1px solid var(--color-border-light);
    border-left: 3px solid var(--color-accent);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    margin-top: 0.75rem;
}
.wa-note-title {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}
.wa-note-text {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--color-text-secondary);
    line-height: 1.5;
    margin: 0;
}

/* ── Year filter pill ── */
.wa-filter-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="margin-bottom:0.25rem;">
        <span class="ag-badge ag-badge-info" style="margin-bottom:0.5rem;">Climate Data</span>
        <h1 style="font-size:1.5rem !important; font-weight:700 !important;
                   margin:0.35rem 0 0 0 !important; padding:0 !important;">
            Weather Analyst
        </h1>
        <p style="font-size:0.84rem; color:var(--color-text-secondary);
                  margin:0.15rem 0 0 0; font-weight:400;">
            Explore historical climate trends and daily weather patterns for agricultural planning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color: str,
    y_suffix: str = "",
    height: int = 280,
) -> go.Figure:
    """Create a compact, transparent-bg Plotly line chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba") if color.startswith("rgb") else f"rgba({_hex_to_rgb(color)}, 0.08)",
            hovertemplate=f"%{{x|%b %d, %Y}}: %{{y:.1f}}{y_suffix}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=10, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", size=11, color="#5A6B60"),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=10, color="#8A9B90"),
            dtick="M3",
            tickformat="%b %Y",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(213,221,217,0.5)",
            gridwidth=1,
            tickfont=dict(family="DM Mono, monospace", size=10, color="#8A9B90"),
            ticksuffix=y_suffix,
            zeroline=False,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_size=12,
            font_family="DM Sans, sans-serif",
        ),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex colour to 'r, g, b' string for rgba()."""
    h = hex_color.lstrip("#")
    return ", ".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_climate():
    """Load and normalise climate CSV with caching."""
    df = load_climate_csv(CLIMATE_CSV)
    if df.empty:
        return df

    # Normalise column names to match the rest of AgroSage
    rename_map = {}
    if "meantemp" in df.columns and "temperature" not in df.columns:
        rename_map["meantemp"] = "temperature"
    if "meanpressure" in df.columns and "pressure" not in df.columns:
        rename_map["meanpressure"] = "pressure"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


climate_df = _load_climate()
has_data = not climate_df.empty and "date" in climate_df.columns


# ═══════════════════════════════════════════════════════════════════════════════
#  BRANCH: DATA AVAILABLE
# ═══════════════════════════════════════════════════════════════════════════════

if has_data:
    # ── Year filtering ────────────────────────────────────────────────────────
    climate_df["year"] = climate_df["date"].dt.year
    available_years = sorted(climate_df["year"].unique().tolist())

    st.markdown(
        '<div class="wa-section"><div class="wa-section-title">Data Filter</div></div>',
        unsafe_allow_html=True,
    )

    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        selected_years = st.multiselect(
            "Filter by Year",
            options=available_years,
            default=available_years,
            help="Select one or more years to display in the charts below.",
            key="wa_year_filter",
        )
    with filter_col2:
        st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
        if st.button("Reset Filter", key="wa_reset_filter", use_container_width=True):
            st.session_state["wa_year_filter"] = available_years
            st.rerun()

    if not selected_years:
        st.warning("Select at least one year to view climate data.")
        st.stop()

    filtered_df = climate_df[climate_df["year"].isin(selected_years)].copy()

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="wa-section"><div class="wa-section-title">Dataset Summary</div></div>',
        unsafe_allow_html=True,
    )

    total_records = len(filtered_df)
    date_range_start = filtered_df["date"].min().strftime("%b %d, %Y")
    date_range_end = filtered_df["date"].max().strftime("%b %d, %Y")

    avg_temp = filtered_df["temperature"].mean() if "temperature" in filtered_df.columns else None
    avg_hum = filtered_df["humidity"].mean() if "humidity" in filtered_df.columns else None
    avg_wind = filtered_df["wind_speed"].mean() if "wind_speed" in filtered_df.columns else None

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="wa-stat">
                <div class="wa-stat-label">Records</div>
                <div class="wa-stat-value">{total_records:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        temp_display = f"{avg_temp:.1f} C" if avg_temp is not None else "—"
        st.markdown(
            f"""
            <div class="wa-stat">
                <div class="wa-stat-label">Avg Temperature</div>
                <div class="wa-stat-value">{temp_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        hum_display = f"{avg_hum:.1f}%" if avg_hum is not None else "—"
        st.markdown(
            f"""
            <div class="wa-stat">
                <div class="wa-stat-label">Avg Humidity</div>
                <div class="wa-stat-value">{hum_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        wind_display = f"{avg_wind:.1f} km/h" if avg_wind is not None else "—"
        st.markdown(
            f"""
            <div class="wa-stat">
                <div class="wa-stat-label">Avg Wind Speed</div>
                <div class="wa-stat-value">{wind_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <p style="font-size:0.76rem; color:var(--color-text-muted); margin-top:0.5rem;">
            Date range: {date_range_start} — {date_range_end}
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Temperature chart ────────────────────────────────────────────────────
    if "temperature" in filtered_df.columns:
        st.markdown(
            '<div class="wa-section"><div class="wa-section-title">Temperature Trend</div></div>',
            unsafe_allow_html=True,
        )
        fig_temp = _build_line_chart(
            filtered_df, "date", "temperature",
            title="Mean Temperature",
            color="#E67E22",
            y_suffix=" C",
        )
        st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})

    # ── Humidity chart ───────────────────────────────────────────────────────
    if "humidity" in filtered_df.columns:
        st.markdown(
            '<div class="wa-section"><div class="wa-section-title">Humidity Trend</div></div>',
            unsafe_allow_html=True,
        )
        fig_hum = _build_line_chart(
            filtered_df, "date", "humidity",
            title="Humidity",
            color="#2980B9",
            y_suffix="%",
        )
        st.plotly_chart(fig_hum, use_container_width=True, config={"displayModeBar": False})

    # ── Wind speed chart ─────────────────────────────────────────────────────
    if "wind_speed" in filtered_df.columns:
        st.markdown(
            '<div class="wa-section"><div class="wa-section-title">Wind Speed Trend</div></div>',
            unsafe_allow_html=True,
        )
        fig_wind = _build_line_chart(
            filtered_df, "date", "wind_speed",
            title="Wind Speed",
            color="#8E44AD",
            y_suffix=" km/h",
        )
        st.plotly_chart(fig_wind, use_container_width=True, config={"displayModeBar": False})

    # ── Monthly averages table ───────────────────────────────────────────────
    st.markdown(
        '<div class="wa-section"><div class="wa-section-title">Monthly Averages</div></div>',
        unsafe_allow_html=True,
    )

    monthly = get_monthly_stats(filtered_df)

    if not monthly.empty:
        # Rename for display
        display_cols = {}
        if "month" in monthly.columns:
            display_cols["month"] = "Month"
        if "avg_temp" in monthly.columns:
            display_cols["avg_temp"] = "Avg Temp (C)"
        if "avg_humidity" in monthly.columns:
            display_cols["avg_humidity"] = "Avg Humidity (%)"
        if "total_rainfall" in monthly.columns:
            display_cols["total_rainfall"] = "Total Rainfall (mm)"
        if "avg_wind_speed" in monthly.columns:
            display_cols["avg_wind_speed"] = "Avg Wind (km/h)"

        display_monthly = monthly.rename(columns=display_cols)

        # Only show columns that actually have data
        valid_cols = [c for c in display_monthly.columns if not display_monthly[c].isna().all()]
        display_monthly = display_monthly[valid_cols]

        st.dataframe(
            display_monthly,
            use_container_width=True,
            hide_index=True,
            height=min(400, len(display_monthly) * 36 + 40),
        )
    else:
        st.info("Monthly aggregation is not available for the selected date range.")

    # ── Data source note ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="wa-note">
            <div class="wa-note-title">Data Source</div>
            <p class="wa-note-text">
                Historical climate data sourced from the DailyDelhiClimate dataset covering
                daily temperature, humidity, wind speed, and atmospheric pressure recordings.
                Use the year filter above to focus on specific time periods. Monthly averages
                are computed from the filtered daily records.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  BRANCH: NO DATA — MOCK WEATHER FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

else:
    st.warning(
        "**Climate dataset not found.** Place `DailyDelhiClimate.csv` in the `data/` folder "
        "to unlock historical analysis. Showing mock weather data below as a preview."
    )

    st.markdown(
        '<div class="wa-section"><div class="wa-section-title">Mock Weather Preview</div></div>',
        unsafe_allow_html=True,
    )

    city_options = [
        "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
        "Hyderabad", "Pune", "Jaipur", "Lucknow",
        "New York", "London", "Tokyo", "Sydney",
    ]

    selected_city = st.selectbox(
        "Select a city for mock weather",
        options=city_options,
        index=0,
        help="These are generated mock values based on typical city climate profiles.",
        key="wa_mock_city",
    )

    mock = get_mock_weather(selected_city)
    current = mock["current"]

    # ── Mock weather card ────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="wa-mock-card">
            <div class="wa-mock-eyebrow">Current Conditions (Mock)</div>
            <div class="wa-mock-city">{mock['city']}</div>
            <div class="wa-mock-climate">{mock['climate_zone']}</div>
            <div class="wa-mock-temp">{current['temperature']} C</div>
            <div class="wa-mock-condition">{current['condition']}</div>
            <div class="wa-mock-details">
                <div class="wa-mock-detail">
                    <span class="wa-mock-detail-label">Feels Like</span>
                    <span class="wa-mock-detail-value">{current['feels_like']} C</span>
                </div>
                <div class="wa-mock-detail">
                    <span class="wa-mock-detail-label">Humidity</span>
                    <span class="wa-mock-detail-value">{current['humidity']}%</span>
                </div>
                <div class="wa-mock-detail">
                    <span class="wa-mock-detail-label">Wind</span>
                    <span class="wa-mock-detail-value">{current['wind_speed']} km/h</span>
                </div>
                <div class="wa-mock-detail">
                    <span class="wa-mock-detail-label">Pressure</span>
                    <span class="wa-mock-detail-value">{current['pressure']} hPa</span>
                </div>
                <div class="wa-mock-detail">
                    <span class="wa-mock-detail-label">Visibility</span>
                    <span class="wa-mock-detail-value">{current['visibility_km']} km</span>
                </div>
                <div class="wa-mock-detail">
                    <span class="wa-mock-detail-label">UV Index</span>
                    <span class="wa-mock-detail-value">{current['uv_index']}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ── 5-day forecast ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="wa-section"><div class="wa-section-title">5-Day Forecast (Mock)</div></div>',
        unsafe_allow_html=True,
    )

    for day in mock["forecast"]:
        st.markdown(
            f"""
            <div class="wa-forecast-row">
                <span class="wa-forecast-day">{day['day_name'][:3]}, {day['date'][5:]}</span>
                <span class="wa-forecast-cond">{day['condition']}</span>
                <span class="wa-forecast-temps">{day['temp_high']} / {day['temp_low']} C</span>
                <span class="wa-forecast-rain">{day['rainfall_chance']}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Mock stat cards ──────────────────────────────────────────────────────
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    ms1, ms2, ms3, ms4 = st.columns(4)

    with ms1:
        st.markdown(
            f"""
            <div class="wa-stat">
                <div class="wa-stat-label">Temperature</div>
                <div class="wa-stat-value">{current['temperature']} C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ms2:
        st.markdown(
            f"""
            <div class="wa-stat">
                <div class="wa-stat-label">Humidity</div>
                <div class="wa-stat-value">{current['humidity']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ms3:
        st.markdown(
            f"""
            <div class="wa-stat">
                <div class="wa-stat-label">Wind Speed</div>
                <div class="wa-stat-value">{current['wind_speed']} km/h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ms4:
        st.markdown(
            f"""
            <div class="wa-stat">
                <div class="wa-stat-label">Pressure</div>
                <div class="wa-stat-value">{current['pressure']} hPa</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="wa-note" style="margin-top:1.25rem;">
            <div class="wa-note-title">Note</div>
            <p class="wa-note-text">
                The data shown above is generated from city-specific climate profiles
                and does not represent real-time observations. To enable historical
                climate analysis with interactive charts, place the
                <code>DailyDelhiClimate.csv</code> dataset in the <code>data/</code> folder.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
