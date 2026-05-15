import os
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j AuraDB ─────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j+s://XXXXXXXX.databases.neo4j.io")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Groq LLM API ─────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
GROQ_MODEL     = "llama3-8b-8192"

# ── OpenRouteService (optional routing) ──────────────────────────────────────
ORS_API_KEY    = os.getenv("ORS_API_KEY",    "")
ORS_BASE_URL   = "https://api.openrouteservice.org/v2/directions/driving-car"

# ── App settings ─────────────────────────────────────────────────────────────
SUPPORTED_CITIES = [
    "Philadelphia", "Nashville", "Tampa", "Indianapolis",
    "Tucson", "Reno", "New Orleans", "Santa Barbara",
]

TOURISM_CATEGORIES = [
    "Restaurants", "Food", "Bars", "Nightlife", "Coffee & Tea",
    "Museums", "Arts & Entertainment", "Hotels & Travel",
    "Shopping", "Attractions & Activities", "Tours",
    "Parks", "Landmarks & Historical Buildings",
]

BUDGET_LABELS = {
    "budget":   "$  (under $15)",
    "moderate": "$$  ($15–$40)",
    "upscale":  "$$$  ($40–$80)",
    "luxury":   "$$$$  ($80+)",
}

DAY_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

PLACES_PER_DAY   = 4
DEFAULT_TOP_N    = 30
HYBRID_WEIGHTS   = {"graph": 0.40, "content": 0.35, "quality": 0.25}
