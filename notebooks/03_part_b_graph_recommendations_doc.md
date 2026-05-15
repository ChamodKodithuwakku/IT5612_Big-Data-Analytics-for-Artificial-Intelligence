# Notebook 03 — Part B: Graph-Based Tourism Recommendation System
### Viva Explanation Guide

**File:** `03_part_b_graph_recommendations.py`  
**What this notebook is:** The third and most complex notebook. It takes the cleaned data from Notebook 01 and the knowledge graph loaded by Notebook 02, then builds a four-method hybrid recommendation system that turns a traveller's stated preferences into a ranked, day-by-day itinerary.

---

## The Core Problem Being Solved

Before explaining the code, it helps to understand why this problem is hard:

- **Cold-start problem** — most tourists have no Yelp review history in the destination city they are visiting. Standard collaborative filtering (which relies on past behaviour) fails completely here.
- **One-time visit problem** — unlike movies or music, most people visit a restaurant or museum once. This makes the user-business rating matrix extremely sparse.
- **Spatial coherence problem** — a list of highly-rated places scattered across a city is not a useful itinerary. Recommendations need to be geographically grouped by day.

The solution is a **four-method hybrid system** that each address these challenges differently.

---

## Section 1 — Environment Setup & Connections

```python
spark = (
    SparkSession.builder
    .appName('TripGraph-PartB')
    .config('spark.driver.memory', '4g')
    .config('spark.sql.shuffle.partitions', '20')
    .getOrCreate()
)
```

**What this is doing:**  
Creates a SparkSession similar to Notebook 01. The shuffle partitions are reduced to 20 here (vs 50 in Notebook 01) because we are working with smaller filtered subsets of data in this notebook — fewer partitions means less scheduling overhead.

---

### Neo4j connection

```python
NEO4J_URI      = 'neo4j+s://040e4230.databases.neo4j.io'
NEO4J_PASSWORD = 'o2ZlvfOSZr0ZP5Qb0sV6M21QZo29JzTJRMK4bXbBSEI'
NEO4J_USER     = '040e4230'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

with driver.session() as session:
    result = session.run('RETURN "Neo4j connected" AS msg').single()
    print(result['msg'])
```

**What this is doing and why:**  
Neo4j is our graph database, running on AuraDB (Neo4j's cloud service). We connect to it using the official `neo4j` Python driver. The URI uses the `neo4j+s://` scheme which means an encrypted Bolt connection. We verify the connection immediately by running a trivial Cypher query — if this fails, all the graph-based algorithms below would fail, so it is important to catch it early.

```python
def run_cypher(query, params=None):
    with driver.session() as session:
        return [dict(r) for r in session.run(query, **(params or {}))]
```

This helper function wraps the Neo4j session boilerplate — it opens a session, runs a Cypher query with optional parameters, converts each result row to a Python dict, and returns a list of dicts. Every Cypher call in this notebook uses this helper.

---

## Section 2 — Data Loading

```python
biz_df     = spark.read.parquet(f'{OUT_DIR}/businesses_clean')
review_df  = spark.read.parquet(f'{OUT_DIR}/reviews_clean')
user_df    = spark.read.parquet(f'{OUT_DIR}/users_clean')
biz_cat_df = spark.read.parquet(f'{OUT_DIR}/biz_categories')

tourism_biz = (
    biz_df
    .filter(F.col('is_tourism_relevant') == True)
    .filter(F.col('city').isin(TARGET_CITIES))
    .filter(F.col('review_count') >= 5)
    .cache()
)
```

**What this is doing and why:**  
We read the four Parquet files produced by Notebook 01. Reading Parquet is much faster than reading JSON because Parquet is a columnar binary format — Spark only reads the columns it needs rather than parsing every field. 

`tourism_biz` is the filtered working dataset for this entire notebook — only tourism-relevant businesses in our 8 target cities with at least 5 reviews. The `review_count >= 5` filter removes very new or obscure businesses that have too little data to rank reliably. `.cache()` pins this in memory because it is used repeatedly.

---

## Section 3 — Method A: Content-Based Filtering (TF-IDF)

### What content-based filtering is
Content-based filtering recommends items that are similar to what the user has said they like — it does not need any data about what other users did. In our case, a user says they like `['Italian', 'Museums', 'Coffee & Tea']` and we find businesses whose category profile closely matches those terms.

### The TF-IDF pipeline

```python
tokenizer   = Tokenizer(inputCol='cat_text', outputCol='words')
remover     = StopWordsRemover(inputCol='words', outputCol='filtered')
hashing_tf  = HashingTF(inputCol='filtered', outputCol='raw_features', numFeatures=512)
idf         = IDF(inputCol='raw_features', outputCol='tfidf_features')
normalizer  = Normalizer(inputCol='tfidf_features', outputCol='norm_features', p=2.0)

content_pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, normalizer])
content_model    = content_pipeline.fit(biz_text)
biz_vectors      = content_model.transform(biz_text)
```

**What each stage does:**

- `Tokenizer` — splits the categories string `"restaurants food coffee tea"` into a list of individual words: `["restaurants", "food", "coffee", "tea"]`. The regex cleanup beforehand (`regexp_replace`) removes special characters like `&` and `+`.

- `StopWordsRemover` — removes common English words that carry no meaning ("the", "and", "of", etc.) from the word list.

- `HashingTF` — converts each filtered word list into a fixed-size numeric vector of dimension 512 using the hashing trick. Each word is hashed to a position 0–511, and its count is placed there. This avoids needing to maintain a vocabulary dictionary.

- `IDF` (Inverse Document Frequency) — downweights terms that appear across many businesses. "Restaurants" appears everywhere, so it gets a low IDF weight — it does not help discriminate between businesses. "Landmarks & Historical Buildings" is rare, so it gets a high IDF weight and carries more discriminating power.

- `Normalizer` (L2) — divides each vector by its L2 norm so all vectors have length 1. This means the dot product between two normalised vectors equals their cosine similarity — making score computation very efficient.

The `Pipeline` chains all stages together. `.fit(biz_text)` runs the fitting step (mainly computing IDF weights from the corpus). `.transform(biz_text)` applies the fitted pipeline to produce a vector for every business.

---

### Making recommendations

```python
def content_based_recommend(user_categories, city, budget=None, top_n=20):
    pref_text = ' '.join(user_categories).lower()
    pref_text = re.sub('[^a-zA-Z ]', ' ', pref_text)

    pref_df  = spark.createDataFrame([(pref_text,)], ['cat_text'])
    pref_vec = content_model.transform(pref_df).select('norm_features').collect()[0][0]
    pref_arr = np.array(pref_vec.toArray()).reshape(1, -1)

    city_mask = biz_vectors_pd['city'] == city
    sims = cosine_similarity(pref_arr, feature_matrix[city_mask.values])[0]
    city_biz['content_score'] = sims

    return city_biz.sort_values('content_score', ascending=False).head(top_n)
```

**What this is doing:**  
We take the user's stated preferences as a text string, pass it through the same fitted pipeline (so it gets the same TF-IDF treatment as the business vectors), and get back a query vector. Then `cosine_similarity()` from scikit-learn computes the similarity between this query vector and every business vector in the target city. The result is a score between 0 and 1 for each business, sorted descending. An optional `budget` parameter filters by `price_label` before sorting.

**Why cosine similarity and not Euclidean distance?**  
Cosine similarity measures the angle between two vectors, not their magnitude. This means it focuses on what categories are mentioned (and how prominently) rather than how long the text is. A business with 10 categories that includes "Museums" and a business with 2 categories that only has "Museums" would have the same relevance to a Museums query — which is the correct behaviour.

---

## Section 4 — Method B: ALS Collaborative Filtering (Spark MLlib)

### What collaborative filtering is
Collaborative filtering recommends items based on what similar users liked — without needing to know anything about the item's content. "Users who reviewed the same businesses as you also liked X" is the core idea.

### ALS — Alternating Least Squares

```python
user_indexer = StringIndexer(inputCol='user_id',     outputCol='user_idx',    handleInvalid='skip')
biz_indexer  = StringIndexer(inputCol='business_id', outputCol='business_idx', handleInvalid='skip')
```

**Why StringIndexer?**  
ALS requires integer user and item IDs — it cannot work with string IDs like Yelp's `user_id` format (e.g., `"xDIjt7..."`). `StringIndexer` assigns a unique integer to each string value. `handleInvalid='skip'` means if a user or business appears in test data but not in training, that row is dropped rather than causing an error.

```python
als = ALS(
    userCol='user_idx',
    itemCol='business_idx',
    ratingCol='stars',
    rank=20,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy='drop',
    seed=42
)
als_model = als.fit(train_als)
```

**What ALS does:**  
ALS factorises the user-business rating matrix into two smaller matrices: a user factor matrix (one 20-dimensional vector per user) and a business factor matrix (one 20-dimensional vector per business). The predicted rating for a user-business pair is the dot product of their respective factor vectors.

"Alternating Least Squares" describes how it solves this: it fixes the business factors and solves for the best user factors (ordinary least squares), then fixes user factors and solves for business factors, and alternates until convergence. `rank=20` means each user and business is represented by a 20-dimensional latent vector. `regParam=0.1` is L2 regularisation to prevent overfitting. `coldStartStrategy='drop'` removes test rows where the user or item was not seen in training (otherwise ALS produces NaN predictions for unknown users).

```python
evaluator = RegressionEvaluator(metricName='rmse', labelCol='stars', predictionCol='prediction')
rmse = evaluator.evaluate(predictions)
mae  = evaluator.setMetricName('mae').evaluate(predictions)
```

**Evaluation:**  
RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) measure how far the predicted star ratings are from the actual ratings on the held-out test set.

### Cold-start handling

```python
def als_recommend_for_user(user_id_str, city, top_n=10):
    user_rows = als_indexed.filter(F.col('user_id') == user_id_str).limit(1).collect()
    if not user_rows:
        print(f'User {user_id_str} not seen in training — cold-start, use content-based.')
        return None
```

**What this is doing and why:**  
If the user has never reviewed anything in the dataset (a new TripGraph user), ALS has no latent vector for them — it simply cannot make predictions. This function detects that case and returns `None`, which signals to the calling code to fall back to Method A (content-based filtering) which requires no history.

---

## Section 5 — Method C: Association Rule Mining (FP-Growth, Spark MLlib)

### What association rule mining is
Association rules find patterns of the form: *"users who visited A also tend to visit B"*. Originally designed for supermarket basket analysis ("customers who buy bread also buy butter"), we apply it at the tourism category level.

### Building user baskets

```python
user_baskets = (
    review_df
    .join(tourism_ids, on='business_id', how='inner')
    .join(biz_cat_df.filter(F.col('category').isin(TOURISM_CATEGORIES)), on='business_id', how='inner')
    .groupBy('user_id')
    .agg(F.collect_set('category').alias('categories_visited'))
    .filter(F.size(F.col('categories_visited')) >= 2)
)
```

**What this is doing:**  
For each user, we collect the set of distinct tourism categories they have reviewed (`collect_set` — not `collect_list` — so each category only appears once per user regardless of how many restaurants they reviewed). We filter to users who visited at least 2 different categories, because a basket of size 1 cannot produce any interesting rules. Each row is now one "transaction" in the FP-Growth sense — a user's category shopping basket.

### Training FP-Growth

```python
fp_growth = FPGrowth(
    itemsCol='categories_visited',
    minSupport=0.01,
    minConfidence=0.30
)
fp_model = fp_growth.fit(user_baskets)
```

**What FP-Growth does:**  
FP-Growth (Frequent Pattern Growth) is an algorithm for finding frequent itemsets — combinations of categories that appear together in at least `minSupport` fraction of baskets. `minSupport=0.01` means a category combination must appear in at least 1% of user baskets to be considered frequent. `minConfidence=0.30` means for a rule `A → B`, at least 30% of users who visited A must also have visited B.

FP-Growth builds a compressed tree structure of the data (the FP-tree) and mines it without generating all candidate itemsets explicitly — this makes it much more efficient than the older Apriori algorithm.

### Understanding the metrics

**Support** = fraction of all baskets that contain the itemset.  
**Confidence** of `A → B` = fraction of baskets containing A that also contain B.  
**Lift** of `A → B` = confidence / (support of B alone). Lift > 1 means the two items co-occur more than pure chance. A lift of 2.3 for `Museums → Coffee & Tea` means museum visitors are 2.3× more likely to visit a coffee shop than the average user.

### Using the rules

```python
def association_rule_recommend(visited_categories):
    visited_set = set(visited_categories)
    suggestions = {}
    for _, row in rules_pd.iterrows():
        if set(row['antecedent']).issubset(visited_set):
            for cat in row['consequent']:
                if cat not in visited_set:
                    suggestions[cat] = suggestions.get(cat, 0) + row['confidence'] * row['lift']
    return sorted(suggestions.items(), key=lambda x: x[1], reverse=True)
```

**What this is doing:**  
Given the user's stated interest categories, this function finds all rules whose antecedent (left-hand side) is a subset of those interests. For each such rule, it suggests the consequent (right-hand side) categories and accumulates a score of `confidence × lift` — so categories suggested by multiple high-confidence, high-lift rules score higher. This is used in the full pipeline to expand the user's preferences before calling the hybrid recommender.

---

## Section 6 — Method D: Graph-Based Hybrid (NetworkX as GDS Fallback)

### Why graph algorithms for recommendations
A graph captures relationships that matrix-based methods miss. If user A and user B both reviewed the same coffee shop, and user B also loved a particular museum, then that museum is "reachable" from user A's interests through the graph — even though user A has never reviewed it. This transitivity is the core advantage of graph-based recommendations.

### Why NetworkX instead of Neo4j GDS
The notebook was designed to use Neo4j GDS (Graph Data Science) for server-side graph algorithms. However, the free tier of AuraDB does not support GDS projections. So we pull the graph data out of Neo4j via Cypher and rebuild the graph locally in NetworkX, then run the same algorithms client-side.

### Building the NetworkX graph

```python
reviewed_rows = run_cypher("""
MATCH (u:User)-[r:REVIEWED]->(b:Business)
RETURN u.user_id AS user_id, b.business_id AS biz_id, r.stars AS stars,
       b.quality_score AS quality_score, b.stars AS biz_stars, b.city AS city
""")

G_nx = nx.DiGraph()
for row in reviewed_rows:
    G_nx.add_node(row['user_id'], node_type='User')
    G_nx.add_node(row['biz_id'],  node_type='Business', ...)
    G_nx.add_edge(row['user_id'], row['biz_id'], stars=row['stars'])
```

**What this is doing:**  
We pull two sets of data from Neo4j: all user-reviewed-business relationships and all business-in-category relationships. We build a directed graph where User nodes have outgoing edges to Business nodes (weighted by star rating) and Business nodes have outgoing edges to Category nodes. Business node attributes (quality score, stars, city) are stored directly on the nodes.

We also precompute two helper dictionaries:
- `biz_reviewers[biz_id]` = set of all user IDs who reviewed that business (used for Jaccard similarity)
- `city_to_biz[city]` = list of business IDs in that city

---

### 6.1 Personalized PageRank

```python
def personalized_pagerank_recommend(city, categories, top_n=15):
    seeds = run_cypher("""
    MATCH (b:Business)-[:LOCATED_IN]->(ci:City {name: $city})
    MATCH (b)-[:IN_CATEGORY]->(c:Category)
    WHERE c.name IN $categories
    RETURN b.business_id AS business_id
    ORDER BY b.quality_score DESC LIMIT 10
    """, {'city': city, 'categories': categories})

    seed_ids = [s['business_id'] for s in seeds if G_nx.has_node(s['business_id'])]
    personalization = {n: 1.0 / len(seed_ids) for n in seed_ids}
    pr_scores = nx.pagerank(G_nx, alpha=0.85, personalization=personalization, max_iter=100)
```

**What PageRank is:**  
PageRank is Google's original web ranking algorithm. In a standard PageRank, a node's score is determined by how many other important nodes link to it. In **Personalized PageRank**, the random walk is biased: instead of randomly jumping to any node, it jumps to one of the seed nodes with probability `(1 - alpha)` = 0.15. This means nodes connected to the seeds through short paths score higher than nodes that are far away in the graph.

**What this is doing:**  
We first query Neo4j for the top-10 quality businesses in the target city that match the user's preferred categories — these become our seed nodes. We give each seed equal weight (`1 / len(seed_ids)`). Then `nx.pagerank()` runs the personalized walk on the entire NetworkX graph with `alpha=0.85` (the standard damping factor). The result is a score for every node — we filter to Business nodes in the target city and sort by score.

**Why this is powerful:**  
If the user says "I like Museums", we seed on Philadelphia's top-rated museums. PageRank then propagates through the graph: any business frequently reviewed by the same users who also review museums will score higher — even if that business is a coffee shop or restaurant. The graph captures "what people who like museums also tend to visit" without needing explicit association rules.

---

### 6.2 Node Similarity (Jaccard on shared reviewers)

```python
biz_pair_counts = defaultdict(int)
for user, data in G_nx.nodes(data=True):
    if data.get('node_type') != 'User':
        continue
    reviewed = [n for n in G_nx.successors(user)
                if G_nx.nodes[n].get('node_type') == 'Business']
    for b1, b2 in combinations(reviewed, 2):
        biz_pair_counts[(min(b1, b2), max(b1, b2))] += 1

for (b1, b2), shared in biz_pair_counts.items():
    if shared < 2:
        continue
    union = len(biz_reviewers[b1]) + len(biz_reviewers[b2]) - shared
    j = shared / union
    similarity_map[b1].append((b2, j))
    similarity_map[b2].append((b1, j))
```

**What Jaccard similarity is:**  
Jaccard similarity between two sets A and B = `|A ∩ B| / |A ∪ B|` — the size of the intersection divided by the size of the union. Here, the sets are the reviewer sets of each business. Two businesses with many shared reviewers are considered similar.

**What this is doing:**  
For each user, we look at all businesses they reviewed and increment a counter for every pair of those businesses. This gives us the intersection size (shared reviewers) for every business pair. We then compute the union size using the formula `|A| + |B| - |A ∩ B|` and divide to get the Jaccard score. Only pairs with at least 2 shared reviewers are kept (to filter noise). The top 5 most similar businesses are stored per business in `similarity_map`.

```python
def node_similarity_recommend(seed_business_id, top_n=10):
    similar = similarity_map.get(seed_business_id, [])
    if similar:
        biz_ids = [b for b, _ in similar[:top_n]]
        ...
    # Fallback: shared categories as proxy for similarity
```

If a business has no Jaccard similarity data (perhaps it has very few reviewers), the function falls back to a Cypher query that counts shared categories as a simpler proxy.

---

### 6.3 Louvain Community Detection

```python
biz_graph = nx.Graph()
biz_graph.add_nodes_from([n for n, d in G_nx.nodes(data=True) if d.get('node_type') == 'Business'])
for (b1, b2), shared in biz_pair_counts.items():
    if shared >= 2:
        biz_graph.add_edge(b1, b2, weight=shared)

communities = nx_comm.louvain_communities(biz_graph, seed=42)
community_id_map = {biz: cid for cid, comm in enumerate(communities) for biz in comm}
```

**What Louvain community detection is:**  
Louvain is a graph partitioning algorithm. It groups nodes into communities by maximising **modularity** — a measure of how densely connected nodes are within communities compared to between communities. A high modularity score means the communities are well-separated.

**What this is doing:**  
We build an undirected business-to-business graph where edge weight = number of shared reviewers (taken from the `biz_pair_counts` dict we already computed for Node Similarity). Then `louvain_communities()` partitions this graph. The result is a list of communities — each community contains businesses that are frequently visited together by overlapping sets of users.

`community_id_map` is a dictionary mapping each business ID to its community number.

```python
def louvain_recommend(city, categories, top_n=15):
    seeds = run_cypher(...)  # top quality businesses matching user's categories
    seed_ids = [s['business_id'] for s in seeds if community_id_map.get(...)]
    target_community = Counter(community_id_map[s] for s in seed_ids).most_common(1)[0][0]
    comm_biz_ids = [b for b, cid in community_id_map.items() if cid == target_community ...]
```

**What this is doing:**  
We find the top quality seed businesses matching the user's preferences, look up which communities they belong to, and take the most common community as the target. Then we recommend the highest-quality businesses from that same community. The idea is that businesses in the same community are naturally co-visited — they represent a coherent "travel cluster" of places that go well together on a trip.

---

### 6.4 Hybrid Scoring

```python
def hybrid_recommend(city, user_categories, budget=None, top_n=20):
    content_df = content_based_recommend(user_categories, city, budget, top_n * 3)
    ppr_df     = personalized_pagerank_recommend(city, user_categories, top_n * 3)

    merged = content_df.merge(ppr_df[['business_id','pagerank_score']], on='business_id', how='left')
    merged['pagerank_score'] = merged['pagerank_score'].fillna(0.0)

    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    merged['norm_content']  = minmax(merged['content_score'])
    merged['norm_pagerank'] = minmax(merged['pagerank_score'])
    merged['norm_quality']  = minmax(merged['quality_score'])

    merged['hybrid_score'] = (
        0.40 * merged['norm_pagerank'] +
        0.35 * merged['norm_content']  +
        0.25 * merged['norm_quality']
    )
```

**What this is doing and why:**  
The hybrid function combines Method A and Method D alongside the pre-computed quality score from Notebook 01.

We generate a candidate pool using content-based filtering (Method A gives us the top 3×N businesses by category match). We then look up their PageRank scores from Method D and merge them in. Businesses that appear in the content pool but were not scored by PageRank (e.g., new businesses with few graph connections) get a PageRank score of 0.

Each score component is min-max normalised independently to [0, 1] so that no single component's scale dominates. The `+ 1e-9` prevents division by zero when all businesses have identical scores. The final hybrid formula is:

```
hybrid_score = 0.40 × norm_pagerank + 0.35 × norm_content + 0.25 × norm_quality
```

The weights prioritise graph connectivity (40%) — the most novel signal — followed by category match (35%) — the most direct preference signal — and quality score (25%) — a baseline credibility check.

---

## Section 7 — Full Pipeline Demo: Prompt → Itinerary

This section demonstrates the complete end-to-end flow.

### Step 1 — Category expansion via association rules

```python
expanded_cats = list(USER_INTENT['categories'])
rule_suggestions = association_rule_recommend(expanded_cats)
for cat, score in rule_suggestions[:3]:
    if cat not in expanded_cats:
        expanded_cats.append(cat)
```

**What this is doing:**  
The user's stated categories are passed to Method C's `association_rule_recommend()`. The top 3 rule-suggested categories that are not already in the list get appended. For example, if the user said `['Museums', 'Arts & Entertainment']`, the rules might suggest adding `'Coffee & Tea'` because museum visitors commonly also visit coffee shops. This enriches the query before feeding it to the hybrid recommender.

### Step 2 — Hybrid recommendation

```python
recommendations = hybrid_recommend(
    city=USER_INTENT['city'],
    user_categories=expanded_cats,
    budget=USER_INTENT['budget'],
    top_n=total_places + 5
)
```

Calls the hybrid function with expanded categories. We request a few extra places as a buffer in case some drop out in the next step.

### Step 3 — Geographic day clustering

```python
coords = recs_with_geo[['lat','lon']].values
kmeans_geo = SKMeans(n_clusters=USER_INTENT['days'], random_state=42, n_init=10)
recs_with_geo['day'] = kmeans_geo.fit_predict(coords) + 1
```

**What this is doing:**  
We fetch the latitude and longitude for each recommended business from Neo4j and apply scikit-learn's K-Means clustering on the 2D coordinate space. With `n_clusters=days`, each cluster corresponds to one day. The result is that geographically close businesses end up on the same day — minimising travel time within a day. `n_init=10` runs K-Means 10 times with different initialisations and picks the best result, improving stability.

Note: this is scikit-learn K-Means (`SKMeans`), not Spark MLlib K-Means — because at this point we have a small Pandas DataFrame (20–30 rows), so single-machine scikit-learn is the right tool.

### Step 4 — Itinerary display

```python
for day in sorted(recs_with_geo['day'].unique()):
    day_places = recs_with_geo[recs_with_geo['day'] == day].sort_values('hybrid_score', ascending=False)
    for i, (_, place) in enumerate(day_places.iterrows(), 1):
        print(f'  {i}. {place["name"][:38]:<38} {stars_str}  score={score_str}  [{place["price_label"]}]')
```

Prints the itinerary day by day, with each day's stops sorted by hybrid score. Also generates a scatter plot map showing business locations coloured by day.

---

## Section 8 — Evaluation: Precision@K, Recall@K, NDCG@K

### The evaluation strategy

```python
review_with_rank = (
    review_df
    .join(tourism_ids, on='business_id', how='inner')
    .withColumn('rank',
        F.row_number().over(
            Window.partitionBy('user_id').orderBy(F.desc('year'), F.desc('month'))
        )
    )
)
holdout = review_with_rank.filter(F.col('rank') == 1).select('user_id','business_id').toPandas()
train_reviews = review_with_rank.filter(F.col('rank') > 1)
```

**What this is doing:**  
For each user, we rank their reviews chronologically (most recent first) using a window function. The most recent review (`rank == 1`) is held out as the ground truth — the business we are trying to predict. All other reviews (`rank > 1`) are used as training history. Only users with at least 5 tourism reviews are included, to ensure there is enough training history to generate meaningful recommendations.

### The metrics

```python
def precision_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    hits  = len(set(top_k) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    hits  = len(set(top_k) & set(relevant))
    return hits / max(len(relevant), 1)

def ndcg_at_k(recommended, relevant, k):
    top_k   = recommended[:k]
    gains   = [1.0 / np.log2(i + 2) if item in relevant else 0.0 for i, item in enumerate(top_k)]
    dcg     = sum(gains)
    ideal   = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / ideal if ideal > 0 else 0.0
```

**Precision@K** — out of the top K recommendations, what fraction is the held-out business? Since we only have one relevant item per user (single-item holdout), this is either 0 or 1/K.

**Recall@K** — did the held-out business appear anywhere in the top K? With one relevant item, this is either 0 or 1.

**NDCG@K** (Normalised Discounted Cumulative Gain) — rewards relevant items appearing near the top of the list. An item at rank 1 contributes `1 / log2(2) = 1.0`, at rank 2 contributes `1 / log2(3) ≈ 0.63`, etc. The score is normalised by the ideal DCG (if the relevant item were at rank 1). NDCG is the most informative metric here because it captures ranking quality, not just whether the item is in the list.

### Methods compared

1. **Popularity baseline** — recommends the 20 most-reviewed businesses in the user's city. No personalisation whatsoever. This is the floor that any sensible method should beat.

2. **ALS collaborative filtering** — the ALS model is re-trained on the training split to prevent data leakage, then generates top-10 recommendations per user.

3. **Graph hybrid** — `hybrid_recommend()` is run for each evaluation user using their historical categories as a proxy for preferences.

---

## Section 9 — Results & Discussion

The key results table compares Precision@K, Recall@K, and NDCG@K at K=5 and K=10 across all three methods. The graph hybrid is expected to outperform ALS on NDCG because it captures transitivity — indirect connections through the graph surface relevant businesses that pure matrix factorisation would miss.

---

## Assignment Criteria Coverage

| Method | Technique | Notebook Section |
|---|---|---|
| A — Content-Based (TF-IDF) | Content-based recommendations | Section 3 |
| B — Collaborative Filtering (ALS) | Collaborative filtering + MLlib | Section 4 |
| C — Association Rules (FP-Growth) | Association-rule-based recommendation | Section 5 |
| D — Graph Hybrid (NetworkX) | Hybrid recommendation | Section 6 |
| Evaluation | Precision@K, Recall@K, NDCG@K vs baseline | Section 8 |

---

## Key Things to Be Able to Explain in a Viva

**"Why four methods instead of just one?"**  
Each method addresses a different weakness. ALS needs user history — cold-start users get content-based instead. Content-based cannot capture transitivity or community behaviour — PageRank and Louvain add that. Association rules expand the preference space beyond what the user explicitly stated. The hybrid combines all signals so no single failure mode dominates.

**"What is the cold-start problem and how do you handle it?"**  
A new user has no review history so collaborative filtering cannot generate a latent vector for them. We detect this in `als_recommend_for_user()` by checking if the user appears in the training index. If not, we fall back to content-based filtering which only needs the user's stated preferences — no history required.

**"Why min-max normalise before combining scores?"**  
PageRank scores are very small (e.g., 0.00003) while content scores might be around 0.7 and quality scores around 0.6. If we combined the raw values, PageRank would contribute essentially nothing. Normalising each component to [0, 1] puts them all on the same scale so the hand-tuned weights (40/35/25) actually control the relative influence.

**"What is lift and why does it matter more than confidence alone?"**  
Confidence of `A → B` = P(B | A). But if B is already very common (e.g., Restaurants), a high confidence could simply reflect that most people visit restaurants — not a meaningful association. Lift = P(B | A) / P(B) — it tells you how much more likely B is when A is present, compared to B's baseline probability. Lift > 1 confirms a genuine association beyond random co-occurrence.

**"Why K-Means for day grouping instead of a routing algorithm?"**  
K-Means on coordinates is a fast approximation that groups geographically proximate stops onto the same day. A proper routing algorithm (like OpenRouteService) would use real travel times and road distances but requires live API calls. For the notebook demo, K-Means is a reasonable and computationally cheap proxy. The full Streamlit app is noted as a future integration point for real travel times.

**"What does NDCG measure that Precision@K does not?"**  
Precision@K only tells you if the relevant item is in the top K — it does not care whether it is ranked 1st or 10th. NDCG gives partial credit based on rank position, so a method that consistently puts the held-out item near the top will score higher even if absolute precision is similar. NDCG is the more informative metric for ranking evaluation.
