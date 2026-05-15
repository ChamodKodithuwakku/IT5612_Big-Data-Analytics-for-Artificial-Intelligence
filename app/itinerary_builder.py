"""
Itinerary builder — clusters recommendations into days and orders stops.
Optional: calls OpenRouteService for real driving ETAs between stops.
"""

import numpy as np
import pandas as pd
import requests
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
from config import ORS_API_KEY, ORS_BASE_URL, PLACES_PER_DAY, DAY_COLORS


# ── Geographic utilities ──────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in kilometres between two lat/lon points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def nearest_neighbour_order(places: pd.DataFrame) -> pd.DataFrame:
    """
    Order stops within a day using a greedy nearest-neighbour tour.
    Starts from the northernmost point (natural 'morning start').
    """
    if len(places) <= 1:
        return places

    df     = places.copy().reset_index(drop=True)
    coords = df[["lat", "lon"]].values
    start  = int(np.argmax(coords[:, 0]))   # northernmost = highest latitude

    visited = [start]
    remaining = list(range(len(df)))
    remaining.remove(start)

    while remaining:
        last = visited[-1]
        dists = [
            haversine_km(coords[last, 0], coords[last, 1], coords[i, 0], coords[i, 1])
            for i in remaining
        ]
        nearest = remaining[int(np.argmin(dists))]
        visited.append(nearest)
        remaining.remove(nearest)

    return df.iloc[visited].reset_index(drop=True)


# ── Day clustering ────────────────────────────────────────────────────────────

def cluster_into_days(recommendations: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Assign a `day` column (1-indexed) to each recommended business using
    K-Means on lat/lon coordinates. Businesses that are geographically close
    end up on the same day.
    """
    df = recommendations.dropna(subset=["lat", "lon"]).copy()

    total_places = days * PLACES_PER_DAY
    df = df.head(total_places)

    if len(df) == 0:
        return df

    n_clusters = min(days, len(df))
    coords     = df[["lat", "lon"]].values

    if n_clusters == 1 or len(df) == 1:
        df["day"] = 1
    else:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["day"] = km.fit_predict(coords) + 1

    return df


# ── OpenRouteService routing ──────────────────────────────────────────────────

def _ors_route(coords_list: list[tuple]) -> list[dict] | None:
    """
    Call ORS Directions API for a sequence of waypoints.
    Returns list of leg dicts {distance_km, duration_min} or None on failure.
    """
    if not ORS_API_KEY or len(coords_list) < 2:
        return None
    try:
        body = {
            "coordinates": [[lon, lat] for lat, lon in coords_list],
            "instructions": False,
        }
        resp = requests.post(
            ORS_BASE_URL,
            json=body,
            headers={
                "Authorization": ORS_API_KEY,
                "Content-Type":  "application/json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        segments = data["routes"][0]["segments"]
        return [
            {
                "distance_km":  round(seg["distance"] / 1000, 2),
                "duration_min": round(seg["duration"] / 60,   1),
            }
            for seg in segments
        ]
    except Exception:
        return None


def _haversine_legs(places: pd.DataFrame) -> list[dict]:
    """Straight-line distance fallback when ORS is unavailable."""
    legs = []
    for i in range(len(places) - 1):
        a = places.iloc[i]
        b = places.iloc[i + 1]
        d = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        # Approximate driving time: 30 km/h average in-city
        legs.append({"distance_km": round(d, 2), "duration_min": round(d / 30 * 60, 0)})
    return legs


# ── Public interface ──────────────────────────────────────────────────────────

def build_itinerary(
    recommendations: pd.DataFrame,
    days: int,
) -> list[dict]:
    """
    Convert a flat DataFrame of recommended businesses into a structured
    day-by-day itinerary.

    Returns a list of day-dicts:
    [
      {
        "day": 1,
        "color": "#e74c3c",
        "places": DataFrame (ordered stops for the day),
        "legs": [{"distance_km": ..., "duration_min": ...}, ...],
        "total_km": float,
        "total_min": float,
      },
      ...
    ]
    """
    clustered = cluster_into_days(recommendations, days)
    if clustered.empty:
        return []

    itinerary = []
    for day_num in sorted(clustered["day"].unique()):
        day_places = clustered[clustered["day"] == day_num].copy()
        day_places  = nearest_neighbour_order(day_places)

        coords_seq = list(zip(day_places["lat"], day_places["lon"]))
        legs       = _ors_route(coords_seq) or _haversine_legs(day_places)

        total_km  = sum(leg["distance_km"]  for leg in legs) if legs else 0
        total_min = sum(leg["duration_min"] for leg in legs) if legs else 0

        itinerary.append({
            "day":       day_num,
            "color":     DAY_COLORS[(day_num - 1) % len(DAY_COLORS)],
            "places":    day_places.reset_index(drop=True),
            "legs":      legs,
            "total_km":  round(total_km,  1),
            "total_min": round(total_min, 0),
        })

    return itinerary


def itinerary_to_text(itinerary: list[dict], city: str) -> str:
    """Export the itinerary as plain text (for download)."""
    lines = [f"TripGraph Itinerary — {city}", "=" * 50, ""]
    for day in itinerary:
        lines.append(f"DAY {day['day']}")
        lines.append("-" * 30)
        for i, (_, place) in enumerate(day["places"].iterrows(), 1):
            lines.append(
                f"  {i}. {place['name']}  "
                f"({place.get('stars', '?'):.1f}★, {place.get('price_label', '')})"
            )
            if i <= len(day["legs"]):
                leg = day["legs"][i - 1]
                lines.append(
                    f"     ↓ {leg['distance_km']} km  (~{int(leg['duration_min'])} min drive)"
                )
        total = f"Day total: {day['total_km']} km  /  ~{int(day['total_min'])} min travel"
        lines += ["", total, ""]
    return "\n".join(lines)
