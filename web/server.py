"""
TripGraph Web API — FastAPI backend for the proper web frontend.

Run from the project root:
    uvicorn web.server:app --reload --host 0.0.0.0 --port 8000

Then open: http://localhost:8000
"""

import os
import sys
import math
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR    = os.path.join(ROOT_DIR, "app")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

sys.path.insert(0, APP_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"))

from config import (
    SUPPORTED_CITIES, TOURISM_CATEGORIES, BUDGET_LABELS,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GROQ_API_KEY, ORS_API_KEY,
    PLACES_PER_DAY,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="TripGraph API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Recommender cache (per credential set, loaded once per server lifetime) ────
_rec_cache: dict = {}
_rec_lock = __import__("threading").Lock()

def _get_recommender(uri: str, user: str, password: str):
    from recommender import TripGraphRecommender
    key = (uri, user, password)
    if key not in _rec_cache:
        with _rec_lock:
            if key not in _rec_cache:
                print(f"[cache] Loading recommender for {uri[:30]}…")
                rec = TripGraphRecommender(uri=uri, user=user, password=password)
                rec.load_businesses()
                print(f"[cache] Loaded {len(rec._biz_df):,} businesses — cached.")
                _rec_cache[key] = rec
    return _rec_cache[key]


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/config")
async def get_config():
    return {
        "cities":        SUPPORTED_CITIES,
        "categories":    TOURISM_CATEGORIES,
        "budget_labels": BUDGET_LABELS,
    }


# ── Pydantic models ───────────────────────────────────────────────────────────

class PlanRequest(BaseModel):
    text:       Optional[str] = None
    city:       Optional[str] = None
    days:       int           = 3
    categories: List[str]     = []
    budget:     Optional[str] = None


class ExportRequest(BaseModel):
    city:      str
    days:      int
    itinerary: list


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/plan")
def plan_trip(req: PlanRequest):  # sync def so FastAPI runs it in a threadpool — keeps event loop free
    import prompt_processor as pp
    import itinerary_builder as ib

    try:
        city           = req.city
        days           = req.days
        categories     = list(req.categories)
        budget         = req.budget
        intent_summary = None

        if req.text:
            intent = pp.parse_intent(req.text)
            if intent:
                if intent.get("city"):
                    city = intent["city"]
                if intent.get("days"):
                    days = int(intent["days"])
                if intent.get("categories"):
                    categories = intent["categories"]
                budget         = intent.get("budget") or budget
                intent_summary = pp.format_intent_summary(intent)

        if not city:
            raise HTTPException(
                status_code=400,
                detail="No supported city detected. Mention one of the supported cities or use the structured form.",
            )

        if not categories:
            categories = ["Restaurants", "Arts & Entertainment"]

        print(f"[plan] Getting recommender for {NEO4J_URI[:30]}...")
        try:
            rec = _get_recommender(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            print(f"[plan] Recommender ready ({len(rec._biz_df):,} businesses cached)")
        except Exception as exc:
            print(f"[plan] Neo4j error: {exc}")
            raise HTTPException(status_code=503, detail=f"Neo4j connection failed: {exc}")

        print(f"[plan] Getting city stats for {city}...")
        stats = rec.city_stats(city)

        print(f"[plan] Running hybrid recommendations (city={city}, categories={categories}, budget={budget})...")
        top_n = days * PLACES_PER_DAY + 5
        recs  = rec.hybrid_recommend(city=city, categories=categories, budget=budget, top_n=top_n)
        print(f"[plan] Got {len(recs)} recommendations")

        if recs.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for {city}. Try different categories or remove the budget filter.",
            )

        itinerary = ib.build_itinerary(recs, days)

        if not itinerary:
            raise HTTPException(
                status_code=500,
                detail="Could not build itinerary — businesses may be missing coordinates.",
            )

        result_days = []
        for day_data in itinerary:
            places_list = []
            for i, (_, row) in enumerate(day_data["places"].iterrows(), 1):
                try:
                    lat_f = float(row.get("lat"))
                    lon_f = float(row.get("lon"))
                    if math.isnan(lat_f) or math.isnan(lon_f):
                        continue
                except (TypeError, ValueError):
                    continue

                places_list.append({
                    "stop":         i,
                    "name":         str(row.get("name", "")),
                    "rating":       round(float(row.get("stars") or 0), 1),
                    "review_count": int(row.get("review_count") or 0),
                    "price":        str(row.get("price_label") or ""),
                    "categories":   str(row.get("categories") or ""),
                    "address":      str(row.get("address") or ""),
                    "lat":          lat_f,
                    "lng":          lon_f,
                    "score":        round(float(row.get("hybrid_score") or 0), 4),
                    "explanation":  rec.explain(row, categories),
                })

            result_days.append({
                "day":       int(day_data["day"]),
                "color":     day_data["color"],
                "places":    places_list,
                "legs": [
                    {
                        "distance_km":  round(float(leg.get("distance_km", 0)), 2),
                        "duration_min": round(float(leg.get("duration_min", 0)), 1),
                    }
                    for leg in (day_data.get("legs") or [])
                ],
                "total_km":  float(day_data.get("total_km", 0)),
                "total_min": float(day_data.get("total_min", 0)),
            })

        return {
            "city":           city,
            "days":           days,
            "categories":     categories,
            "budget":         budget,
            "intent_summary": intent_summary,
            "stats":          stats,
            "itinerary":      result_days,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[plan] Unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/export")
def export_itinerary(req: ExportRequest):
    lines = [f"TripGraph Itinerary — {req.city}", "=" * 50, ""]
    for day in req.itinerary:
        lines.append(f"DAY {day['day']}")
        lines.append("-" * 30)
        for i, place in enumerate(day["places"], 1):
            stars = place.get("rating") or 0
            price = place.get("price", "")
            lines.append(f"  {i}. {place['name']}  ({stars:.1f}★, {price})")
            legs = day.get("legs", [])
            if i <= len(legs):
                leg = legs[i - 1]
                lines.append(f"     ↓ {leg['distance_km']} km  (~{int(leg['duration_min'])} min drive)")
        lines += ["", f"Day total: {day['total_km']} km  /  ~{int(day['total_min'])} min travel", ""]

    city_slug = req.city.lower().replace(" ", "_")
    return PlainTextResponse(
        content="\n".join(lines),
        headers={
            "Content-Disposition": f'attachment; filename="tripgraph_{city_slug}_{req.days}days.txt"'
        },
    )
