"""
Prompt processing layer.
Primary  : Groq API (Llama 3) — parses natural language into structured JSON.
Fallback : regex + keyword rules — used when Groq key is missing or API fails.
"""

import json
import re
from config import GROQ_API_KEY, GROQ_MODEL, SUPPORTED_CITIES, TOURISM_CATEGORIES


# ── Groq primary parser ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a travel intent parser. Extract structured trip information from the user's message.

Return ONLY a valid JSON object with these fields:
{
  "city":       string (one of the supported cities, or null if not mentioned),
  "days":       integer (number of trip days, default 2 if not mentioned),
  "categories": list of strings (tourism interest categories),
  "budget":     string (one of: "budget", "moderate", "upscale", "luxury", or null),
  "raw_city":   string (the city name as the user wrote it, even if unsupported)
}

Supported cities: """ + ", ".join(SUPPORTED_CITIES) + """

Category mapping:
- food/eat/restaurant/dining/italian/sushi/bbq/seafood → "Restaurants"
- coffee/cafe/brunch → "Coffee & Tea"
- bar/pub/drinks/nightlife/club → "Bars", "Nightlife"
- museum/history/art gallery → "Museums", "Arts & Entertainment"
- hotel/stay/accommodation → "Hotels & Travel"
- shopping/mall/boutique → "Shopping"
- park/outdoor/nature/hiking → "Parks"
- tour/sightseeing/landmark → "Tours", "Landmarks & Historical Buildings"

Budget mapping:
- cheap/budget/affordable/inexpensive → "budget"
- moderate/mid-range/reasonable → "moderate"
- upscale/fancy/nice → "upscale"
- luxury/high-end/fine dining → "luxury"

Return only the JSON object, no explanation."""


def parse_with_groq(prompt_text: str) -> dict | None:
    """Call Groq API to parse the prompt. Returns dict or None on failure."""
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": prompt_text},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if Groq wrapped the JSON
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception:
        return None


# ── Rule-based fallback ───────────────────────────────────────────────────────

_CITY_ALIASES = {
    "philly":         "Philadelphia",
    "new orleans":    "New Orleans",
    "nola":           "New Orleans",
    "santa barbara":  "Santa Barbara",
}

_CATEGORY_KEYWORDS = {
    "Restaurants":                   r"restaurant|food|eat|dine|dining|italian|sushi|bbq|burger|pizza|seafood|mexican|chinese|thai|indian",
    "Coffee & Tea":                  r"coffee|cafe|café|tea|brunch|bakery",
    "Bars":                          r"\bbar\b|pub|brewery|beer|cocktail",
    "Nightlife":                     r"nightlife|club|nightclub|party|live music",
    "Museums":                       r"museum|history|historical|exhibit|gallery",
    "Arts & Entertainment":          r"art|entertainment|theater|theatre|concert|show|comedy",
    "Hotels & Travel":               r"hotel|accommodation|stay|inn|motel",
    "Shopping":                      r"shop|shopping|mall|boutique|market|store",
    "Parks":                         r"park|garden|nature|outdoor|hike|trail|beach",
    "Tours":                         r"tour|sightseeing|walking tour|boat tour",
    "Landmarks & Historical Buildings": r"landmark|monument|historic|famous|iconic",
    "Attractions & Activities":      r"attraction|activity|adventure|sport|amusement",
}

_BUDGET_KEYWORDS = {
    "budget":   r"budget|cheap|affordable|inexpensive|low.cost",
    "moderate": r"moderate|mid.range|reasonable|average",
    "upscale":  r"upscale|fancy|nice|good restaurant",
    "luxury":   r"luxury|luxurious|high.end|fine dining|expensive|splurge",
}


def parse_with_rules(prompt_text: str) -> dict:
    """Rule-based fallback parser using regex."""
    text_lower = prompt_text.lower()

    # City detection
    city = None
    for alias, canonical in _CITY_ALIASES.items():
        if alias in text_lower:
            city = canonical
            break
    if city is None:
        for c in SUPPORTED_CITIES:
            if c.lower() in text_lower:
                city = c
                break

    # Days detection
    days = 2
    day_match = re.search(r"(\d+)[\s-]*day", text_lower)
    if day_match:
        days = min(int(day_match.group(1)), 7)
    elif "weekend" in text_lower:
        days = 2
    elif "week" in text_lower:
        days = 5

    # Category detection
    categories = []
    for cat, pattern in _CATEGORY_KEYWORDS.items():
        if re.search(pattern, text_lower):
            categories.append(cat)
    if not categories:
        categories = ["Restaurants", "Arts & Entertainment"]

    # Budget detection
    budget = None
    for label, pattern in _BUDGET_KEYWORDS.items():
        if re.search(pattern, text_lower):
            budget = label
            break

    return {
        "city":       city,
        "days":       days,
        "categories": categories,
        "budget":     budget,
        "raw_city":   city,
    }


# ── Public interface ──────────────────────────────────────────────────────────

def parse_intent(prompt_text: str) -> dict:
    """
    Parse a natural language travel prompt into structured intent.
    Tries Groq first; falls back to rules if unavailable or on error.
    Returns dict with keys: city, days, categories, budget, raw_city, method.
    """
    result = parse_with_groq(prompt_text)
    if result and isinstance(result, dict):
        result["method"] = "groq"
        # Validate city
        if result.get("city") and result["city"] not in SUPPORTED_CITIES:
            result["city"] = None
        # Ensure categories list is non-empty
        if not result.get("categories"):
            result["categories"] = ["Restaurants", "Arts & Entertainment"]
        result.setdefault("days", 2)
    else:
        result = parse_with_rules(prompt_text)
        result["method"] = "rules"

    return result


def format_intent_summary(intent: dict) -> str:
    """Human-readable summary of parsed intent for display in the UI."""
    city   = intent.get("city") or "Not detected"
    days   = intent.get("days", 2)
    cats   = ", ".join(intent.get("categories", []))
    budget = intent.get("budget") or "any"
    method = intent.get("method", "rules")
    return (
        f"**Parsed intent** _(via {method})_\n"
        f"- City: **{city}**\n"
        f"- Days: **{days}**\n"
        f"- Interests: {cats}\n"
        f"- Budget: **{budget}**"
    )
