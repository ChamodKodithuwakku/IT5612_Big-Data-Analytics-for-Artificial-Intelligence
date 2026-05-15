"""
TripGraph — AI-Powered Travel Itinerary Planner
Streamlit web application.

Run locally:
    cd app && streamlit run main.py

Run on Colab:
    See run_on_colab.ipynb in the project root.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from config import (
    SUPPORTED_CITIES, TOURISM_CATEGORIES, BUDGET_LABELS,
    DAY_COLORS, PLACES_PER_DAY,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GROQ_API_KEY, ORS_API_KEY,
)
from prompt_processor  import parse_intent, format_intent_summary
from recommender        import TripGraphRecommender
from itinerary_builder  import build_itinerary, itinerary_to_text


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TripGraph — AI Travel Planner",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Place card */
.place-card {
    background: #f8f9fa;
    border-left: 4px solid #3498db;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 10px;
}
.place-card h4 { margin: 0 0 4px 0; font-size: 1rem; color: #2c3e50; }
.place-card p  { margin: 2px 0;     font-size: 0.85rem; color: #555; }
.place-number  { font-size: 1.3rem; font-weight: bold; color: #3498db; margin-right: 6px; }
.star-badge    { color: #f39c12; font-weight: bold; }
.budget-badge  { background: #eaf4fb; border-radius: 4px; padding: 1px 6px; font-size: 0.78rem; }
.leg-info      { color: #7f8c8d; font-size: 0.8rem; margin: 2px 0 6px 20px; }
/* Day header */
.day-header {
    padding: 6px 12px;
    border-radius: 6px;
    color: white;
    font-weight: bold;
    font-size: 1.05rem;
    margin: 16px 0 8px 0;
}
/* Metric override */
[data-testid="metric-container"] { background: #f0f4f8; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f5fa.svg", width=48)
    st.title("TripGraph")
    st.caption("AI-powered travel itinerary planner")
    st.divider()

    st.subheader("🔑 API Credentials")
    neo4j_uri  = st.text_input("Neo4j URI",      value=NEO4J_URI,      type="password")
    neo4j_pass = st.text_input("Neo4j Password", value=NEO4J_PASSWORD,  type="password")
    groq_key   = st.text_input("Groq API Key",   value=GROQ_API_KEY,    type="password")
    ors_key    = st.text_input("ORS API Key (optional routing)", value=ORS_API_KEY, type="password")

    # Override config with sidebar values at runtime
    os.environ["NEO4J_URI"]      = neo4j_uri
    os.environ["NEO4J_PASSWORD"] = neo4j_pass
    os.environ["GROQ_API_KEY"]   = groq_key
    os.environ["ORS_API_KEY"]    = ors_key

    st.divider()
    st.subheader("ℹ️ About")
    st.markdown("""
**TripGraph** uses a Neo4j knowledge graph built from **6M+ Yelp reviews** to generate personalised travel itineraries.

**Methods:**
- 🧠 LLM intent parsing (Groq)
- 📄 TF-IDF content matching
- 🕸️ Graph PageRank (Neo4j GDS)
- 📍 K-Means geo clustering
""")


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to Neo4j…")
def get_recommender(uri, password):
    """One recommender instance per session — loads business cache once."""
    rec = TripGraphRecommender(uri=uri, user=NEO4J_USER, password=password)
    rec.load_businesses()
    return rec


# ── Map builder ───────────────────────────────────────────────────────────────

def build_map(itinerary: list[dict], city: str) -> folium.Map:
    """Build a Folium map with day-coloured markers and route polylines."""
    # Centre map on mean coordinates of all places
    all_places = pd.concat([d["places"] for d in itinerary], ignore_index=True)
    centre_lat = all_places["lat"].mean()
    centre_lon = all_places["lon"].mean()

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13, tiles="CartoDB positron")

    for day_data in itinerary:
        color   = day_data["color"]
        places  = day_data["places"]
        day_num = day_data["day"]
        coords  = []

        for i, (_, place) in enumerate(places.iterrows(), 1):
            if pd.isna(place["lat"]) or pd.isna(place["lon"]):
                continue

            lat, lon = place["lat"], place["lon"]
            coords.append([lat, lon])

            popup_html = f"""
            <div style='font-family:sans-serif; min-width:160px'>
              <b style='color:{color}'>{place['name']}</b><br>
              ⭐ {place.get('stars', '?'):.1f} &nbsp; 💬 {int(place.get('review_count', 0)):,} reviews<br>
              💰 {place.get('price_label','')}&nbsp;
              📅 Day {day_num} — Stop {i}
            </div>"""

            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="white",
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"Day {day_num} #{i}: {place['name'][:28]}",
            ).add_to(m)

            # Stop number label
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:9px;font-weight:bold;color:white;text-align:center;margin-top:1px">{i}</div>',
                    icon_size=(18, 18),
                    icon_anchor=(9, 9),
                ),
            ).add_to(m)

        # Route polyline for the day
        if len(coords) > 1:
            folium.PolyLine(
                coords, color=color, weight=2.5, opacity=0.7, dash_array="5"
            ).add_to(m)

    return m


# ── Itinerary display ─────────────────────────────────────────────────────────

def render_itinerary(itinerary: list[dict], recommender: TripGraphRecommender, user_categories: list[str]):
    """Render day cards and map side by side."""
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📋 Day-by-Day Itinerary")
        for day_data in itinerary:
            color   = day_data["color"]
            day_num = day_data["day"]
            places  = day_data["places"]
            total_km  = day_data["total_km"]
            total_min = day_data["total_min"]

            st.markdown(
                f'<div class="day-header" style="background:{color}">📍 Day {day_num} '
                f'— {len(places)} stops &nbsp;|&nbsp; ~{total_km} km &nbsp;|&nbsp; ~{int(total_min)} min travel</div>',
                unsafe_allow_html=True,
            )

            for i, (_, place) in enumerate(places.iterrows(), 1):
                stars   = place.get("stars", 0)
                n_rev   = int(place.get("review_count", 0))
                price   = place.get("price_label", "")
                score   = place.get("hybrid_score", 0)

                st.markdown(f"""
<div class="place-card">
  <h4><span class="place-number">{i}.</span> {place['name']}</h4>
  <p><span class="star-badge">{'★' * round(stars)}</span> {stars:.1f} &nbsp;
     <span class="budget-badge">{price}</span> &nbsp;
     💬 {n_rev:,} reviews &nbsp; 🎯 score {score:.3f}
  </p>
</div>""", unsafe_allow_html=True)

                with st.expander(f"Why this place? — {place['name'][:30]}"):
                    st.markdown(recommender.explain(place, user_categories))

                # Travel leg to next stop
                legs = day_data.get("legs", [])
                if i <= len(legs):
                    leg = legs[i - 1]
                    st.markdown(
                        f'<p class="leg-info">⬇ &nbsp;{leg["distance_km"]} km drive · ~{int(leg["duration_min"])} min</p>',
                        unsafe_allow_html=True,
                    )

    with col_right:
        st.subheader("🗺️ Interactive Map")
        fmap = build_map(itinerary, "")
        st_folium(fmap, width=680, height=560, returned_objects=[])


# ── Main app ──────────────────────────────────────────────────────────────────

st.title("🗺️ TripGraph — AI Travel Itinerary Planner")
st.caption("Powered by Neo4j · Groq · PySpark · Yelp Open Dataset (6M+ reviews)")

# Connect to Neo4j
try:
    recommender = get_recommender(neo4j_uri, neo4j_pass)
    st.success(f"✅ Connected to Neo4j — {len(recommender._biz_df):,} businesses loaded", icon="✅")
except Exception as e:
    st.error(f"❌ Could not connect to Neo4j: {e}")
    st.info("Enter your Neo4j URI and password in the sidebar, then refresh.")
    st.stop()

st.divider()

# ── Input tabs ────────────────────────────────────────────────────────────────

tab_nl, tab_form = st.tabs(["💬 Natural Language", "🔧 Structured Form"])

intent = None

with tab_nl:
    st.markdown("#### Describe your trip in plain English")
    example = '"Plan a 3-day trip to Philadelphia. I love Italian food, museums, and cozy coffee shops. Moderate budget."'
    st.caption(f"Example: {example}")

    prompt = st.text_area(
        "Your travel prompt",
        height=100,
        placeholder="e.g. 3 days in Nashville, I want live music, BBQ, and some outdoor walks. Budget-friendly.",
        label_visibility="collapsed",
    )

    if st.button("✨ Plan My Trip", key="btn_nl", use_container_width=True, type="primary"):
        if not prompt.strip():
            st.warning("Please enter a travel prompt.")
        else:
            with st.spinner("Parsing your prompt…"):
                intent = parse_intent(prompt)

            if not intent.get("city"):
                st.warning(
                    f"Could not detect a supported city. Detected: **{intent.get('raw_city', 'none')}**. "
                    f"Supported: {', '.join(SUPPORTED_CITIES)}."
                )
                intent = None
            else:
                with st.expander("🔍 Parsed intent", expanded=False):
                    st.markdown(format_intent_summary(intent))

with tab_form:
    st.markdown("#### Set your preferences manually")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sel_city = st.selectbox("Destination city", SUPPORTED_CITIES)
    with c2:
        sel_days = st.slider("Trip length (days)", 1, 7, 3)
    with c3:
        sel_budget = st.selectbox("Budget", ["(any)"] + list(BUDGET_LABELS.keys()),
                                  format_func=lambda x: BUDGET_LABELS.get(x, x))

    sel_cats = st.multiselect(
        "Interests / activity types",
        TOURISM_CATEGORIES,
        default=["Restaurants", "Arts & Entertainment", "Coffee & Tea"],
    )

    if st.button("✨ Plan My Trip", key="btn_form", use_container_width=True, type="primary"):
        if not sel_cats:
            st.warning("Select at least one interest.")
        else:
            intent = {
                "city":       sel_city,
                "days":       sel_days,
                "categories": sel_cats,
                "budget":     None if sel_budget == "(any)" else sel_budget,
                "method":     "form",
            }

# ── Run recommendations & display ─────────────────────────────────────────────

if intent:
    city       = intent["city"]
    days       = intent.get("days", 2)
    categories = intent.get("categories", ["Restaurants"])
    budget     = intent.get("budget")

    st.divider()
    st.subheader(f"🏙️ {city} — {days}-Day Itinerary")

    # City stats
    stats = recommender.city_stats(city)
    m1, m2, m3 = st.columns(3)
    m1.metric("Businesses in graph", f"{int(stats.get('total_businesses', 0)):,}")
    m2.metric("Avg rating",          f"{stats.get('avg_rating', 0):.2f} ★")
    m3.metric("Total reviews",       f"{int(stats.get('total_reviews', 0)):,}")

    st.divider()

    # Generate recommendations
    with st.spinner(f"Building your {days}-day {city} itinerary…"):
        recs = recommender.hybrid_recommend(
            city=city,
            categories=categories,
            budget=budget,
            top_n=days * PLACES_PER_DAY + 5,
        )

    if recs.empty:
        st.warning(
            f"No recommendations found for **{city}** with the selected filters. "
            "Try removing the budget filter or selecting different categories."
        )
    else:
        # Build itinerary
        itinerary = build_itinerary(recs, days)

        if not itinerary:
            st.error("Could not build itinerary — businesses may be missing coordinates.")
        else:
            render_itinerary(itinerary, recommender, categories)

            st.divider()

            # Summary stats
            total_stops = sum(len(d["places"]) for d in itinerary)
            total_km    = sum(d["total_km"]  for d in itinerary)
            total_min   = sum(d["total_min"] for d in itinerary)

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total stops",     total_stops)
            sc2.metric("Total km driven", f"{total_km:.1f}")
            sc3.metric("Total travel",    f"~{int(total_min)} min")
            sc4.metric("Days planned",    days)

            # Download
            st.divider()
            text_export = itinerary_to_text(itinerary, city)
            st.download_button(
                label="📥 Download itinerary (.txt)",
                data=text_export,
                file_name=f"tripgraph_{city.lower().replace(' ','_')}_{days}days.txt",
                mime="text/plain",
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "TripGraph · IT5612 Big Data Analytics · "
    "Dataset: [Yelp Open Dataset](https://www.yelp.com/dataset) · "
    "Graph: Neo4j AuraDB · LLM: Groq Llama 3"
)
