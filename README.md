# TripGraph — AI-Powered Travel Itinerary Planner

IT5612 Big Data Analytics for Artificial Intelligence — Assignment

Combines Apache Spark big data analytics (Part A), a Neo4j knowledge graph with PageRank & Node Similarity recommendations (Part B), and an LLM-powered web app that builds personalised multi-day travel itineraries from natural language prompts.

---

## Project Structure

```
TripGraph/
├── notebooks/
│   ├── 01_part_a_pyspark_analytics.ipynb   # Part A — PySpark analysis of Yelp dataset
│   ├── 02_data_to_neo4j.ipynb              # Data pipeline: Spark output → Neo4j graph
│   └── 03_part_b_graph_recommendations.ipynb  # Part B — Graph algorithms & recommendations
├── app/
│   ├── main.py                             # Streamlit web app (entry point)
│   ├── config.py                           # Configuration & API keys
│   ├── prompt_processor.py                 # Groq LLM intent parser
│   ├── recommender.py                      # Neo4j hybrid recommendation engine
│   └── itinerary_builder.py               # Clustering + route optimisation
├── web/
│   ├── server.py                           # FastAPI backend (alternative to Streamlit)
│   └── static/                            # HTML/CSS/JS frontend
├── run_on_colab.ipynb                      # Google Colab launcher
├── requirements.txt                        # Python dependencies
├── .env.example                            # Credential template
└── .env                                    # Your credentials (never commit this)
```

---

## Prerequisites

### API Credentials

You need three services (all free tiers):

| Service | Purpose | Get it at |
|---|---|---|
| **Neo4j AuraDB** | Graph database | [console.neo4j.io](https://console.neo4j.io) — create a free AuraDB instance |
| **Groq** | LLM intent parsing (Llama 3) | [console.groq.com](https://console.groq.com) — free API key |
| **OpenRouteService** *(optional)* | Driving ETAs on the map | [openrouteservice.org](https://openrouteservice.org) — free tier |

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```ini
# .env
NEO4J_URI=neo4j+s://XXXXXXXX.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-aura-password-here
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXX
ORS_API_KEY=                          # leave blank if not using routing
```

---

## Way 1 — Run Locally

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **PySpark note:** PySpark requires Java 8 or 11.  
> Download from [adoptium.net](https://adoptium.net) and set `JAVA_HOME` if it is not already set.

### 2. Run the notebooks in order

Open Jupyter in the project root:

```bash
jupyter notebook
```

Run the three notebooks **in sequence**:

| # | Notebook | What it does |
|---|---|---|
| 1 | `notebooks/01_part_a_pyspark_analytics.ipynb` | Loads the Yelp JSON dataset with PySpark, performs EDA, sentiment analysis, and writes cleaned Parquet files |
| 2 | `notebooks/02_data_to_neo4j.ipynb` | Reads the Parquet output and loads nodes + relationships into Neo4j AuraDB |
| 3 | `notebooks/03_part_b_graph_recommendations.ipynb` | Runs Personalized PageRank, Node Similarity, and community detection; evaluates Precision\@K and Recall\@K |

> Notebook 2 and 3 require your Neo4j credentials in `.env` before running.

### 3a. Launch the Streamlit app

```bash
cd app
streamlit run main.py
```

The app opens automatically at **http://localhost:8501**.

Type a natural language trip request (e.g. *"3-day trip to Philadelphia with Italian food and museums, moderate budget"*) and the app will generate a day-by-day itinerary with an interactive map.

### 3b. Launch the FastAPI server (alternative)

```bash
cd web
uvicorn server:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser to use the HTML/JS frontend backed by the same recommendation engine.

---

## Way 2 — Run on Google Colab

Use this method if you do not have a local Python environment or want a free GPU/compute session.

### Step 1 — Upload the project to Google Drive

Upload the entire `TripGraph/` folder to your Google Drive at:

```
MyDrive/TripGraph/
```

### Step 2 — Add credentials as Colab Secrets

In any Colab notebook go to **Runtime → Secrets (🔑)** and add the following keys:

| Key | Value |
|---|---|
| `NEO4J_URI` | Your AuraDB connection URI |
| `NEO4J_PASSWORD` | Your AuraDB password |
| `GROQ_API_KEY` | Your Groq API key |
| `ORS_API_KEY` | Your ORS key (or leave blank) |

### Step 3 — Open and run `run_on_colab.ipynb`

Open `run_on_colab.ipynb` in Google Colab and **Run all cells** in order. The notebook will:

1. Mount your Google Drive
2. Install all Python dependencies and `localtunnel`
3. Load credentials from Colab Secrets
4. Write a `.env` file for the app
5. Start Streamlit on port 8501 in the background
6. Open a public tunnel via localtunnel and print the URL

When the tunnel URL appears, click it. If prompted for a password, enter the **Colab public IP** that was printed in Step 5 of the notebook.

### Running the notebooks on Colab

To run the analytics notebooks on Colab, open each one individually and add a cell at the top to install PySpark:

```python
!pip install -q pyspark==3.5.0
```

Then mount Drive and set the dataset path to wherever you uploaded the Yelp JSON files on Drive.

---

## Notebook Run Order Summary

```
01_part_a_pyspark_analytics.ipynb   →   writes  data/processed/*.parquet
        ↓
02_data_to_neo4j.ipynb              →   loads   Neo4j AuraDB graph
        ↓
03_part_b_graph_recommendations.ipynb  (reads graph, runs algorithms)
        ↓
app/main.py  or  web/server.py      (reads graph at runtime)
```

---

## Troubleshooting

**`JAVA_HOME is not set` when running PySpark locally**  
Install JDK 11 from [adoptium.net](https://adoptium.net) and set the environment variable:
```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-11..."
```

**`Neo4j connection refused` or authentication error**  
Verify the URI and password in your `.env` file. AuraDB URIs start with `neo4j+s://`.

**Groq LLM times out / returns empty intent**  
The app falls back to regex/NLTK parsing automatically. Check your `GROQ_API_KEY` in `.env`.

**Streamlit port already in use**  
```bash
streamlit run main.py --server.port 8502
```

**localtunnel asks for a password (Colab)**  
Enter the Colab public IP printed in Step 5 of `run_on_colab.ipynb`.
