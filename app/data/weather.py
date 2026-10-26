"""Kickoff weather for NFL games from Open-Meteo (free, no key), keyed by nflverse stadium id."""
import threading
import time
from datetime import datetime, timezone

import httpx

from ..config import OPEN_METEO

# stadium_id -> (lat, lon, roof) where roof is outdoors | dome | retractable
STADIUMS = {
    "ATL97": (33.7554, -84.4008, "retractable"), "BAL00": (39.2780, -76.6227, "outdoors"),
    "BOS00": (42.0909, -71.2643, "outdoors"), "BUF00": (42.7738, -78.7870, "outdoors"),
    "CAR00": (35.2258, -80.8528, "outdoors"), "CHI98": (41.8623, -87.6167, "outdoors"),
    "CIN00": (39.0955, -84.5161, "outdoors"), "CLE00": (41.5061, -81.6995, "outdoors"),
    "DAL00": (32.7473, -97.0945, "retractable"), "DEN00": (39.7439, -105.0201, "outdoors"),
    "DET00": (42.3400, -83.0456, "dome"), "GNB00": (44.5013, -88.0622, "outdoors"),
    "HOU00": (29.6847, -95.4107, "retractable"), "IND00": (39.7601, -86.1639, "retractable"),
    "JAX00": (30.3239, -81.6373, "outdoors"), "KAN00": (39.0489, -94.4839, "outdoors"),
    "LAX01": (33.9535, -118.3392, "dome"), "MIA00": (25.9580, -80.2389, "outdoors"),
    "MIN01": (44.9737, -93.2577, "dome"), "NAS00": (36.1665, -86.7713, "outdoors"),
    "NOR00": (29.9511, -90.0812, "dome"), "NYC01": (40.8135, -74.0745, "outdoors"),
    "PHI00": (39.9008, -75.1675, "outdoors"), "PHO00": (33.5276, -112.2626, "retractable"),
    "PIT00": (40.4468, -80.0158, "outdoors"), "SEA00": (47.5952, -122.3316, "outdoors"),
    "SFO01": (37.4030, -121.9700, "outdoors"), "TAM00": (27.9759, -82.5033, "outdoors"),
    "VEG00": (36.0909, -115.1833, "dome"), "WAS00": (38.9077, -76.8645, "outdoors"),
    # International venues
    "LON00": (51.5560, -0.2796, "outdoors"), "LON02": (51.6043, -0.0664, "outdoors"),
    "MEX00": (19.3029, -99.1505, "outdoors"), "MUN01": (48.2188, 11.6247, "outdoors"),
    "MAD01": (40.4531, -3.6883, "retractable"), "GER00": (50.0686, 8.6455, "outdoors"),
    "SAO00": (-23.5453, -46.4742, "outdoors"), "RIO00": (-22.9122, -43.2302, "outdoors"),
    "MEL00": (-37.8200, 144.9834, "outdoors"), "PAR00": (48.9245, 2.3602, "outdoors"),
}

# From nflverse 2010-2026 outdoor games: actual totals fall ~0.3 pts per mph of wind above 10.
WIND_TOTAL_PER_MPH = -0.3
WIND_FREE_MPH = 10.0

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _hourly(lat: float, lon: float) -> dict | None:
    key = f"{lat},{lon}"
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 3600:
            return hit[1]
    try:
        with httpx.Client(timeout=20) as client:
            r = client.get(OPEN_METEO, params={
                "latitude": lat, "longitude": lon, "timezone": "UTC", "forecast_days": 10,
                "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability,precipitation,snowfall",
                "wind_speed_unit": "mph", "temperature_unit": "fahrenheit", "precipitation_unit": "inch",
            })
            r.raise_for_status()
            data = r.json()["hourly"]
    except (httpx.HTTPError, KeyError):
        return None
    with _lock:
        _cache[key] = (time.time(), data)
    return data


def prefetch(stadium_id: str) -> None:
    lat, lon, roof = STADIUMS.get(stadium_id, (None, None, "dome"))
    if lat is not None and roof != "dome":
        _hourly(lat, lon)


def forecast(stadium_id: str, kickoff: datetime) -> dict:
    """Weather at kickoff plus the model's total adjustment. Unknown stadiums return roof only."""
    lat, lon, roof = STADIUMS.get(stadium_id, (None, None, "unknown"))
    out = {"roof": roof, "total_adj": 0.0}
    if roof == "dome" or lat is None:
        return out
    hourly = _hourly(lat, lon)
    if not hourly:
        return out
    target = kickoff.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00")
    if target not in hourly["time"]:
        return out  # beyond the forecast horizon
    i = hourly["time"].index(target)
    wx = {
        "temp_f": hourly["temperature_2m"][i], "wind_mph": hourly["wind_speed_10m"][i],
        "gust_mph": hourly["wind_gusts_10m"][i], "precip_prob": hourly["precipitation_probability"][i],
        "precip_in": hourly["precipitation"][i], "snow_in": hourly["snowfall"][i],
    }
    out.update(wx)
    if roof == "outdoors" and wx["wind_mph"] is not None:
        out["total_adj"] = WIND_TOTAL_PER_MPH * max(0.0, wx["wind_mph"] - WIND_FREE_MPH)
    return out


def describe(wx: dict) -> str | None:
    if wx.get("roof") == "dome":
        return "Dome"
    if "wind_mph" not in wx:
        return "Retractable roof" if wx.get("roof") == "retractable" else None
    s = f"{wx['temp_f']:.0f}°F, wind {wx['wind_mph']:.0f} mph"
    if (wx.get("precip_prob") or 0) >= 30:
        s += f", {'snow' if (wx.get('snow_in') or 0) > 0 else 'precip'} {wx['precip_prob']:.0f}%"
    if wx["roof"] == "retractable":
        s += " (retractable roof, likely closed in bad weather)"
    return s
