"""
AgroSage — Weather & Climate Utility Functions
Provides CSV climate loading, monthly aggregation, and mock weather fallback.
"""

import os
import random
import datetime
import pandas as pd
import numpy as np


# ─── Climate CSV Loader ───────────────────────────────────────────────────────

def load_climate_csv(path: str) -> pd.DataFrame:
    """
    Load a climate / weather CSV file and return a cleaned DataFrame.

    Expected columns (flexible — works with any superset):
        date, temperature, humidity, rainfall, wind_speed

    Returns an empty DataFrame with correct columns if the file is missing
    or unreadable, so downstream code never crashes.
    """
    _FALLBACK_COLS = [
        "date", "temperature", "humidity",
        "rainfall", "wind_speed", "pressure",
    ]

    if not os.path.isfile(path):
        return pd.DataFrame(columns=_FALLBACK_COLS)

    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except (ValueError, KeyError):
        # If 'date' column doesn't exist, try generic parse
        try:
            df = pd.read_csv(path)
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df.rename(columns={col: "date"}, inplace=True)
                    break
        except Exception:
            return pd.DataFrame(columns=_FALLBACK_COLS)
    except Exception:
        return pd.DataFrame(columns=_FALLBACK_COLS)

    # Ensure numeric columns
    numeric_cols = ["temperature", "humidity", "rainfall", "wind_speed", "pressure"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─── Monthly Aggregation ─────────────────────────────────────────────────────

def get_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a climate DataFrame by month.

    Returns a DataFrame with columns:
        month, avg_temp, avg_humidity, total_rainfall, avg_wind_speed

    If the input is empty or lacks a 'date' column, returns an empty DataFrame.
    """
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=[
            "month", "avg_temp", "avg_humidity",
            "total_rainfall", "avg_wind_speed",
        ])

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)

    agg_map = {}
    if "temperature" in df.columns:
        agg_map["temperature"] = "mean"
    if "humidity" in df.columns:
        agg_map["humidity"] = "mean"
    if "rainfall" in df.columns:
        agg_map["rainfall"] = "sum"
    if "wind_speed" in df.columns:
        agg_map["wind_speed"] = "mean"

    if not agg_map:
        return pd.DataFrame(columns=[
            "month", "avg_temp", "avg_humidity",
            "total_rainfall", "avg_wind_speed",
        ])

    monthly = df.groupby("month").agg(agg_map).reset_index()

    rename_map = {
        "temperature": "avg_temp",
        "humidity": "avg_humidity",
        "rainfall": "total_rainfall",
        "wind_speed": "avg_wind_speed",
    }
    monthly.rename(columns=rename_map, inplace=True)

    # Round for readability
    for col in ["avg_temp", "avg_humidity", "total_rainfall", "avg_wind_speed"]:
        if col in monthly.columns:
            monthly[col] = monthly[col].round(2)

    return monthly


# ─── Mock Weather Generator ──────────────────────────────────────────────────

_CITY_PROFILES = {
    "mumbai":       {"temp": (26, 34), "hum": (65, 90), "rain": (0, 45), "wind": (8, 22),  "desc": "Tropical coastal"},
    "delhi":        {"temp": (12, 42), "hum": (30, 75), "rain": (0, 30), "wind": (5, 18),  "desc": "Semi-arid continental"},
    "bangalore":    {"temp": (18, 32), "hum": (45, 80), "rain": (0, 25), "wind": (6, 16),  "desc": "Tropical savanna"},
    "chennai":      {"temp": (24, 38), "hum": (60, 85), "rain": (0, 35), "wind": (8, 20),  "desc": "Tropical wet-dry"},
    "kolkata":      {"temp": (16, 36), "hum": (55, 90), "rain": (0, 40), "wind": (5, 15),  "desc": "Tropical wet-dry"},
    "hyderabad":    {"temp": (18, 38), "hum": (35, 75), "rain": (0, 20), "wind": (6, 16),  "desc": "Semi-arid"},
    "pune":         {"temp": (16, 36), "hum": (40, 80), "rain": (0, 30), "wind": (5, 18),  "desc": "Tropical wet-dry"},
    "jaipur":       {"temp": (10, 44), "hum": (20, 60), "rain": (0, 15), "wind": (6, 20),  "desc": "Semi-arid desert"},
    "ahmedabad":    {"temp": (14, 42), "hum": (25, 70), "rain": (0, 20), "wind": (6, 18),  "desc": "Semi-arid"},
    "lucknow":      {"temp": (10, 42), "hum": (35, 80), "rain": (0, 25), "wind": (4, 14),  "desc": "Humid subtropical"},
    "new york":     {"temp": (-2, 32), "hum": (40, 75), "rain": (0, 20), "wind": (8, 25),  "desc": "Humid subtropical"},
    "london":       {"temp": (2, 24),  "hum": (60, 90), "rain": (0, 15), "wind": (10, 28), "desc": "Oceanic temperate"},
    "tokyo":        {"temp": (2, 34),  "hum": (45, 85), "rain": (0, 25), "wind": (6, 18),  "desc": "Humid subtropical"},
    "sydney":       {"temp": (10, 28), "hum": (45, 75), "rain": (0, 18), "wind": (8, 22),  "desc": "Oceanic temperate"},
    "default":      {"temp": (10, 35), "hum": (40, 80), "rain": (0, 20), "wind": (5, 20),  "desc": "Temperate"},
}

_WEATHER_CONDITIONS = [
    ("☀️ Clear Sky",       "clear"),
    ("⛅ Partly Cloudy",   "partly_cloudy"),
    ("☁️ Overcast",        "overcast"),
    ("🌧️ Light Rain",     "light_rain"),
    ("🌦️ Showers",        "showers"),
    ("⛈️ Thunderstorm",   "thunderstorm"),
    ("🌫️ Foggy",          "foggy"),
    ("💨 Windy",           "windy"),
]


def get_mock_weather(city: str) -> dict:
    """
    Generate realistic mock weather data for a given city.

    Returns a dict with:
        city, climate_zone, current (temp, humidity, wind, pressure, condition,
        condition_icon, feels_like, visibility, uv_index),
        forecast (list of 5 daily dicts), timestamp.

    Uses city-specific climate profiles when available, otherwise falls back
    to a generic temperate profile.
    """
    key = city.strip().lower()
    profile = _CITY_PROFILES.get(key, _CITY_PROFILES["default"])

    temp_lo, temp_hi = profile["temp"]
    hum_lo, hum_hi = profile["hum"]
    rain_lo, rain_hi = profile["rain"]
    wind_lo, wind_hi = profile["wind"]

    # Current conditions
    current_temp = round(random.uniform(temp_lo, temp_hi), 1)
    current_hum = random.randint(hum_lo, hum_hi)
    current_wind = round(random.uniform(wind_lo, wind_hi), 1)
    current_pressure = random.randint(1005, 1025)
    condition_label, condition_code = random.choice(_WEATHER_CONDITIONS)
    feels_like = round(current_temp + random.uniform(-3, 2), 1)
    visibility = round(random.uniform(4, 15), 1)
    uv_index = random.randint(1, 11)

    # 5-day forecast
    today = datetime.date.today()
    forecast = []
    for i in range(1, 6):
        day = today + datetime.timedelta(days=i)
        day_temp_hi = round(random.uniform(current_temp - 3, current_temp + 5), 1)
        day_temp_lo = round(day_temp_hi - random.uniform(4, 10), 1)
        day_cond_label, day_cond_code = random.choice(_WEATHER_CONDITIONS)
        forecast.append({
            "date": day.isoformat(),
            "day_name": day.strftime("%A"),
            "temp_high": day_temp_hi,
            "temp_low": day_temp_lo,
            "humidity": random.randint(hum_lo, hum_hi),
            "wind_speed": round(random.uniform(wind_lo, wind_hi), 1),
            "rainfall_chance": random.randint(0, 100),
            "condition": day_cond_label,
            "condition_code": day_cond_code,
        })

    return {
        "city": city.strip().title(),
        "climate_zone": profile["desc"],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "current": {
            "temperature": current_temp,
            "humidity": current_hum,
            "wind_speed": current_wind,
            "pressure": current_pressure,
            "condition": condition_label,
            "condition_code": condition_code,
            "feels_like": feels_like,
            "visibility_km": visibility,
            "uv_index": uv_index,
        },
        "forecast": forecast,
    }
