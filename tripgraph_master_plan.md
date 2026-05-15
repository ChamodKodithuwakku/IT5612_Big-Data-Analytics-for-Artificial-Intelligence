# TripGraph Master Plan

## Project: AI-Powered Travel Itinerary Planner

A natural language trip planner powered by a graph-based recommendation engine, trained on 6M+ Yelp reviews using PySpark.

## 1. Executive Summary

| | |
|---|---|
| **Project Name** | TripGraph — AI Travel Itinerary Planner |
| **Assignment Coverage** | Part A + Part B (bonus marks) |
| **Big Data Component** | Yelp Open Dataset (6M+ reviews, 150K+ businesses) |
| **Core Innovation** | Graph-based recommendation + LLM intent extraction + Live trip routing |
| **Deliverable** | Working web application + analytics notebook |

## 2. System Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│              (Streamlit / Flask web app)                 │
│       "Plan a 3-day trip to Philadelphia, I like         │
│        Italian food and museums, moderate budget"        │
└─────────────────────────┬────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│              PROMPT PROCESSING LAYER                     │
│  ┌────────────────┐         ┌────────────────────────┐  │
│  │  Primary:      │  fail   │  Fallback:             │  │
│  │  Groq LLM API  │ ──────→ │  spaCy + Rule-based    │  │
│  └────────────────┘         └────────────────────────┘  │
│              Output: Structured JSON intent              │
└─────────────────────────┬────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│           GRAPH RECOMMENDATION ENGINE                    │
│                  (Neo4j AuraDB)                          │
│                                                          │
│  Built from PySpark-processed Yelp data:                 │
│  - Personalized PageRank                                 │
│  - Node Similarity                                       │
│  - Community Detection                                   │
│  Returns: ranked business IDs with scores                │
└─────────────────────────┬────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│              ITINERARY OPTIMIZER                         │
│   - Cluster places by location                           │
│   - Order by minimum travel time                         │
│   - OpenRouteService API for routing & ETA               │
└─────────────────────────┬────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│                  RESULT DISPLAY                          │
│   - Day-by-day itinerary cards                           │
│   - Interactive Leaflet map with route                   │
│   - Travel time between stops                            │
│   - "Why this place" explanation per recommendation      │
└──────────────────────────────────────────────────────────┘
```

## 3. Assignment Mapping

### Part A — Big Data Analytics (PySpark)

| Task | Implementation |
|---|---|
| Dataset loading | Yelp JSON files loaded as Spark DataFrames |
| Data cleaning | Null handling, schema enforcement, deduplication |
| Transformation | Flatten nested JSON, parse categories, geo features |
| EDA | Business density by city, category distribution, rating patterns |
| Analytics | Sentiment analysis on reviews, elite user behaviour, temporal trends |
| Visualization | Matplotlib, Seaborn, Plotly for interactive charts |
| Insights | Business insights report with key findings |

### Part B — Recommendation System (Graph-Based)

| Task | Implementation |
|---|---|
| Method | Knowledge Graph + Graph Algorithms (Neo4j GDS) |
| Data prep | Spark transforms data → loaded into Neo4j |
| Graph schema | Users, Businesses, Categories, Cities, Friends |
| Algorithms | Personalized PageRank, Node Similarity, Louvain Communities |
| Evaluation | Precision@K, Recall@K, comparison with baseline (popularity) |
| Demo | Live web app — user prompt → recommendations |

## 4. Tech Stack

| Layer | Technology |
|---|---|
| **Big Data Processing** | Apache Spark / PySpark |
| **Graph Database** | Neo4j AuraDB (free tier) |
| **Graph Algorithms** | Neo4j Graph Data Science library |
| **LLM (Primary)** | Groq API (Llama 3) |
| **NLP (Fallback)** | spaCy + NLTK |
| **Routing** | OpenRouteService API |
| **Maps** | Leaflet.js + OpenStreetMap |
| **Frontend** | Streamlit (fastest) or Flask + HTML |
| **Notebook** | Jupyter / Google Colab |
| **Language** | Python 3.10+ |

**Total cost: $0** — all free tiers

## 5. Dataset

| | |
|---|---|
| **Source** | Yelp Open Dataset |
| **URL** | https://www.yelp.com/dataset |
| **Mirror** | Kaggle — yelp-dataset |
| **Size** | ~10 GB uncompressed |
| **Files** | business.json, review.json, user.json, checkin.json, tip.json |
| **Coverage** | 150K+ businesses across 11 metro areas |
| **License** | Free for academic & personal projects |

## 6. Module Breakdown

```text
trip-planner/
│
├── notebooks/
│   ├── 01_part_a_pyspark_analytics.ipynb     ← Part A deliverable
│   ├── 02_data_to_neo4j.ipynb                ← Pipeline
│   └── 03_part_b_graph_recommendations.ipynb ← Part B deliverable
│
├── app/
│   ├── prompt_processor.py    ← LLM + NLP fallback
│   ├── recommender.py         ← Neo4j query layer
│   ├── itinerary_builder.py   ← Routing + optimization
│   ├── main.py                ← Streamlit app
│   └── config.py              ← API keys, settings
│
├── data/
│   └── yelp/                  ← raw Yelp JSON files
│
├── requirements.txt
├── README.md
└── demo_video.mp4
```

## 7. Build Roadmap

| Phase | Goal | Output |
|---|---|---|
| **Phase 1** | Setup environment | Spark, Neo4j AuraDB, Groq API keys ready |
| **Phase 2** | Part A — PySpark analytics | Notebook with EDA, insights, visualizations |
| **Phase 3** | Build the graph | Yelp data → Neo4j with full schema |
| **Phase 4** | Recommendation engine | Working graph queries returning top-N places |
| **Phase 5** | Prompt processor | LLM + NLP fallback module |
| **Phase 6** | Itinerary builder | Routing + travel time optimization |
| **Phase 7** | Web frontend | Streamlit app tying it all together |
| **Phase 8** | Polish & evaluate | Precision/Recall metrics, edge cases |
| **Phase 9** | Submission prep | Slides, video, README |

## 8. Submission Deliverables Checklist

- [ ] Python notebooks (Part A + Part B implementations)
- [ ] Source code (`app/` folder with all modules)
- [ ] Dataset link (Yelp Open Dataset URL in README)
- [ ] Presentation slides (max 10 slides)
- [ ] Demo video (showing prompt → recommendations → itinerary)
- [ ] README (setup, run instructions, architecture)
- [ ] requirements.txt (all Python dependencies)

## 9. Presentation Slide Plan (10 slides max)

| # | Slide |
|---|---|
| 1 | Title + team |
| 2 | Problem statement & motivation |
| 3 | System architecture diagram |
| 4 | Dataset overview (Yelp) |
| 5 | Part A — PySpark analytics highlights |
| 6 | Part B — Graph schema & recommendation approach |
| 7 | Prompt processing — LLM + NLP fallback |
| 8 | Demo screenshots / live demo |
| 9 | Evaluation metrics & key findings |
| 10 | Conclusion & future work |

## 10. Demo Video Script (~3-5 mins)

1. Intro (15s) — what the system does
2. Architecture walkthrough (45s) — show the diagram
3. Part A notebook (60s) — scroll through key analytics
4. Neo4j graph (45s) — show the actual knowledge graph
5. Live web demo (90s) — type prompt → show recommendation → show map
6. Wrap-up (15s) — outcomes & impact
