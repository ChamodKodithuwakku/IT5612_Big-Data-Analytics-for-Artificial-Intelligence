# Notebook 01 — Part A: Big Data Analytics Using Apache Spark
### Viva Explanation Guide

**File:** `01_part_a_pyspark_analytics.py`  
**What this notebook is:** The first of three notebooks. Its job is to take the raw Yelp dataset — over 150,000 businesses and 6 million reviews — and use Apache Spark to clean, enrich, and analyse it. Everything it produces gets saved as files that the later notebooks use to build the recommendation system.

---

## Section 1 — Environment Setup & Spark Initialisation

### Setting up the environment

```python
!apt-get install openjdk-11-jdk-headless -qq > /dev/null
!pip install -q pyspark==3.5.0 textblob nltk plotly seaborn
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
```

**What this is doing and why:**  
Apache Spark is a Java-based framework — it runs on the JVM (Java Virtual Machine). So before anything else, we install Java 11 on the Colab machine. Then we install PySpark (the Python interface to Spark) along with the libraries we will use for visualisation and sentiment analysis. The `JAVA_HOME` line tells PySpark exactly where Java is installed so it can start the Spark engine.

---

### Downloading the Yelp Dataset

```python
ZIP_URL  = 'https://business.yelp.com/external-assets/files/Yelp-JSON.zip'
if not os.path.exists(ZIP_PATH):
    !wget --no-verbose --show-progress -O "{ZIP_PATH}" "{ZIP_URL}"
```

**What this is doing and why:**  
The Yelp Open Dataset is roughly 4 GB as a zip file and about 10 GB when extracted. We download it directly from Yelp's website using `wget`. The `if not os.path.exists` check means we only download it once — if the zip is already there (e.g., from a previous run), we skip the download. After downloading, we extract the zip to get a `.tar` archive, then extract the tar to get the actual JSON files. We then walk the directory tree to automatically detect where the JSON files landed and set `DATA_DIR`.

---

### Starting Spark

```python
spark = (
    SparkSession.builder
    .appName('TripGraph-PartA')
    .config('spark.driver.memory', '4g')
    .config('spark.sql.shuffle.partitions', '50')
    .getOrCreate()
)
```

**What this is doing and why:**  
This creates the Spark engine — called a `SparkSession`. Think of it as turning on the distributed computing system. The key configurations here are:
- `spark.driver.memory = 4g` — gives the main Spark process 4 GB of RAM so it does not crash when collecting large results
- `spark.sql.shuffle.partitions = 50` — when Spark needs to redistribute data across workers (e.g., during a `groupBy`), it uses 50 partitions. The default of 200 is too many for a single-machine Colab environment and wastes overhead

---

### Constants defined here

```python
TOURISM_CATEGORIES = ['Restaurants', 'Food', 'Bars', 'Nightlife', 'Coffee & Tea',
                      'Museums', 'Arts & Entertainment', 'Hotels & Travel', ...]

TARGET_CITIES = ['Philadelphia', 'Nashville', 'Tampa', 'Indianapolis', ...]
```

**What these are and why:**  
`TOURISM_CATEGORIES` is a list of 13 category labels that are travel-relevant. We use this throughout the notebook to filter the dataset down to businesses a tourist would actually care about — ignoring things like car mechanics or dentists. `TARGET_CITIES` is a list of 8 cities that are well-represented in the Yelp dataset. We focus on these cities so our analysis has enough data density to produce meaningful results.

---

## Section 2 — Dataset Loading & Initial Inspection

```python
df_business = spark.read.json(BUSINESS_PATH)
df_review   = spark.read.json(REVIEW_PATH)
df_user     = spark.read.json(USER_PATH)
df_checkin  = spark.read.json(CHECKIN_PATH)
df_tip      = spark.read.json(TIP_PATH)
```

**What this is doing and why:**  
The Yelp dataset comes as five separate JSON files, each representing a different entity. We load all five into Spark DataFrames. Spark reads these in a distributed way — it does not load the entire file into RAM at once. Instead, it splits the data into partitions and processes them in parallel. `spark.read.json()` automatically infers the schema (column names and types) by scanning the data.

The five files are:
- **business** — ~150K businesses with name, location, stars, categories, attributes
- **review** — ~6M reviews linking users to businesses with star ratings and text
- **user** — ~1.9M user profiles with review count, fans, elite status
- **checkin** — timestamps of when people checked into each business (foot traffic indicator)
- **tip** — short comments users leave, shorter than full reviews

After loading, we inspect row and column counts, print schemas, show sample rows, and count null values in critical fields. This step is important because it tells us the shape of the data and reveals any quality problems before we start cleaning.

---

## Section 3 — Data Cleaning, Preprocessing & Transformation

### Business cleaning

```python
biz_clean = (
    df_business
    .dropna(subset=['business_id', 'name', 'city', 'state', 'stars', 'categories'])
    .dropDuplicates(['business_id'])
    .filter(F.col('is_open') == 1)
    .withColumn('stars',        F.col('stars').cast(FloatType()))
    .withColumn('review_count', F.col('review_count').cast(IntegerType()))
    .withColumn('price_range',
        F.col('attributes.RestaurantsPriceRange2').cast(IntegerType()))
    .withColumn('has_delivery',
        F.when(F.col('attributes.RestaurantsDelivery') == 'True', True).otherwise(False))
    .withColumn('lat_bucket',  F.round(F.col('latitude'),  3))
    .withColumn('lon_bucket',  F.round(F.col('longitude'), 3))
    .drop('attributes', 'hours')
)
```

**What this is doing and why:**  
This is a method-chaining pipeline — each step returns a new DataFrame with the transformation applied. In Spark this is lazy, meaning nothing actually runs until we call an action like `.count()` or `.show()`. Here is what each step does:

- `dropna(subset=[...])` — removes any row where one of these essential fields is missing. A business without a name, city, or categories is useless for our analysis.
- `dropDuplicates(['business_id'])` — removes any business that appears more than once. business_id should be a unique key.
- `filter(is_open == 1)` — keeps only currently open businesses. Recommending a closed restaurant to a tourist is a bad experience.
- `.cast()` calls — the JSON parser sometimes reads numbers as strings. We explicitly cast `stars` to a float and `review_count` to an integer so we can do arithmetic on them.
- `price_range` — the Yelp dataset stores restaurant attributes in a nested struct. We pull out the price range (1=budget, 4=luxury) by accessing `attributes.RestaurantsPriceRange2`.
- `has_delivery` / `outdoor_seating` — similarly extracted from the nested struct. These become useful filter options in the web app.
- `lat_bucket` / `lon_bucket` — rounding coordinates to 3 decimal places gives a spatial grid cell (~100 metre precision), useful for grouping nearby businesses.
- `.drop('attributes', 'hours')` — removes the original nested columns after we have extracted what we need. This reduces memory usage.

---

### Category explosion

```python
biz_exploded = (
    biz_clean
    .withColumn('category_array', F.split(F.trim(F.col('categories')), ',\\s*'))
    .withColumn('category', F.explode(F.col('category_array')))
    .withColumn('category', F.trim(F.col('category')))
    .filter(F.col('category') != '')
)
```

**What this is doing and why:**  
In the raw Yelp data, a business's categories are stored as a single comma-separated string — for example `"Restaurants, Food, Coffee & Tea"`. To do any category-level analysis (like counting how many businesses are in each category, or building association rules), we need one row per category. 

`F.split()` converts the string into an array: `["Restaurants", "Food", "Coffee & Tea"]`. Then `F.explode()` turns that array into separate rows, so one business becomes three rows — one per category. `F.trim()` cleans up any leading/trailing whitespace that the split might leave behind.

---

### Review cleaning

```python
review_clean = (
    df_review
    .dropna(subset=['review_id','user_id','business_id','stars','text','date'])
    .dropDuplicates(['review_id'])
    .withColumn('stars', F.col('stars').cast(FloatType()))
    .withColumn('date',  F.to_timestamp(F.col('date'), 'yyyy-MM-dd HH:mm:ss'))
    .withColumn('year',  F.year(F.col('date')))
    .withColumn('month', F.month(F.col('date')))
    .filter(F.length(F.col('text')) > 20)
)
```

**What this is doing and why:**  
Similar to business cleaning. The key additions here are:
- Parsing the `date` string into a proper Spark timestamp using the format `'yyyy-MM-dd HH:mm:ss'`. Once it is a timestamp, we can extract `year` and `month` as integer columns — these are used for temporal trend analysis in Section 6.
- `filter(length(text) > 20)` — removes very short reviews that are too brief to carry useful sentiment information (e.g., a review that just says "nice!").

---

### User cleaning

```python
user_clean = (
    df_user
    ...
    .withColumn('is_elite',
        F.when((F.col('elite').isNotNull()) & (F.col('elite') != ''), True)
         .otherwise(False))
)
```

**What this is doing and why:**  
The `elite` field in the Yelp dataset is a comma-separated string of years in which the user held Yelp Elite status (e.g., `"2018,2019,2020"`). Rather than parse the years, we just create a boolean `is_elite` flag — True if the field is non-null and non-empty, False otherwise. Elite users are important because they write significantly more reviews and have more followers, making them high-signal nodes in the TripGraph knowledge graph.

---

### Checkin aggregation

```python
checkin_counts = (
    df_checkin
    .withColumn('checkin_count', F.size(F.split(F.col('date'), ', ')))
    .select('business_id', 'checkin_count')
)
```

**What this is doing and why:**  
The checkin `date` field is a single string of comma-separated timestamps like `"2018-01-01 14:00:00, 2018-01-03 19:00:00, ..."`. Each timestamp represents one check-in event. Rather than parsing individual timestamps, we just count how many commas there are using `F.split().size()`. This gives us a `checkin_count` per business — a proxy for physical foot traffic volume that supplements the review count.

---

## Section 4 — Feature Engineering

### Joining checkins and filling defaults

```python
biz_enriched = biz_clean.join(checkin_counts, on='business_id', how='left')
biz_enriched = biz_enriched.fillna({'checkin_count': 0, 'price_range': 2})
```

**What this is doing and why:**  
We left-join because not every business has checkin data — a left join keeps all businesses and puts null for the checkin count when there is no match. Then `fillna` replaces those nulls: missing checkin count becomes 0 (no recorded checkins), and missing price range defaults to 2 (moderate) because that is the most common tier.

---

### Price label

```python
biz_enriched = biz_enriched.withColumn(
    'price_label',
    F.when(F.col('price_range') == 1, 'budget')
     .when(F.col('price_range') == 2, 'moderate')
     .when(F.col('price_range') == 3, 'upscale')
     .when(F.col('price_range') == 4, 'luxury')
     .otherwise('unknown')
)
```

**What this is doing and why:**  
The raw `price_range` is an integer 1–4. This converts it to a human-readable string. `F.when().when().otherwise()` is Spark's equivalent of a SQL CASE WHEN statement. The string label is what gets stored in Neo4j and shown to users in the web app filter panel.

---

### Log review count

```python
biz_enriched = biz_enriched.withColumn(
    'log_review_count', F.log1p(F.col('review_count').cast(FloatType()))
)
```

**What this is doing and why:**  
Review counts are heavily right-skewed — most businesses have fewer than 100 reviews, but a few popular places have tens of thousands. If we used the raw count in a composite score, those few outliers would dominate everything. `log1p(x)` computes `log(1 + x)`, which compresses the large values: a business with 10,000 reviews gets a log score of ~9.2, while one with 100 reviews gets ~4.6 — a much more moderate difference than the raw 100:1 ratio.

---

### Tourism relevance flag

```python
tourism_pattern = '|'.join(TOURISM_CATEGORIES)
biz_enriched = biz_enriched.withColumn(
    'is_tourism_relevant',
    F.col('categories').rlike(tourism_pattern)
)
```

**What this is doing and why:**  
`'|'.join(TOURISM_CATEGORIES)` builds a regex pattern like `"Restaurants|Food|Bars|Nightlife|..."`. Then `rlike()` checks if the business's categories string matches any of these patterns. This creates a boolean flag `is_tourism_relevant` that we use throughout the notebook to filter down to travel-meaningful businesses without re-filtering on category every time.

---

### City-level normalisation

```python
city_window = Window.partitionBy('city')

biz_enriched = (
    biz_enriched
    .withColumn('city_min_stars', F.min('stars').over(city_window))
    .withColumn('city_max_stars', F.max('stars').over(city_window))
    .withColumn('norm_stars',
        (F.col('stars') - F.col('city_min_stars')) /
        (F.col('city_max_stars') - F.col('city_min_stars') + F.lit(1e-6))
    )
    ...
)
```

**What this is doing and why:**  
This is a window function — a Spark feature that computes an aggregate (like min or max) over a group of rows without collapsing those rows the way `groupBy` does. `Window.partitionBy('city')` means "compute the min/max within each city separately". 

The reason we normalise per city is that raw star ratings are not comparable across cities — one city might have higher average ratings than another due to cultural differences in reviewing behaviour. By min-max normalising within each city, we make a 4.0-star business in Philadelphia directly comparable to a 4.0-star business in Nashville. The `+ 1e-6` epsilon prevents division by zero when a city has only one business (where min equals max).

---

### Preliminary quality score

```python
biz_enriched = biz_enriched.withColumn(
    'quality_score_prelim',
    F.round(0.6 * F.col('norm_stars') + 0.4 * F.col('norm_log_rev'), 4)
)
```

**What this is doing and why:**  
This is a first-pass quality score that combines stars (60%) and review volume (40%) before we add sentiment. It is "preliminary" because sentiment analysis (Section 6) will refine it into the final `quality_score`. The weights (0.6 and 0.4) reflect that star rating is a more direct quality signal than volume, but volume gives us confidence that the rating is based on many people's opinions rather than just a few.

`biz_enriched.cache()` is called after this — caching tells Spark to keep this DataFrame in memory so it does not recompute everything from the raw JSON files each time we use it.

---

## Section 5 — Exploratory Data Analysis & Visualisations

All visualisations use a helper function:
```python
def to_pandas(spark_df):
    return spark_df.toPandas()
```
This collects a Spark DataFrame from the distributed system into a regular Pandas DataFrame on the driver node so matplotlib/seaborn can plot it.

### 5.1 Business Density by City
Groups businesses by city, counts them, takes the top 20, and plots a horizontal bar chart. This shows which cities have enough data to support reliable recommendations.

### 5.2 Top Tourism Categories
Uses `biz_exploded` (the category-exploded table) to count businesses per category. Shows that Restaurants dominate the dataset at ~40%.

### 5.3 Star Rating Distribution
Two bar charts — one for all businesses, one for tourism businesses only. Shows the well-known Yelp rating bias: ratings cluster heavily at 3.5–4.5 stars. Very few businesses get 1 or 2 stars because unhappy customers often do not bother reviewing. This is why we cannot rely on raw stars alone for ranking.

### 5.4 Review Volume Over Time
Groups reviews by year and month and plots a time series. Shows steady growth through the 2010s, a sharp drop during COVID-19 (2020–2021 is highlighted with a red shaded band), and partial recovery. This validates our decision to use all-time history rather than recent data only.

### 5.5 Elite vs Non-Elite User Behaviour
Groups users by `is_elite` and computes average reviews written, average fans, and average stars given. Elite users write far more reviews and have more followers — confirming they are high-signal nodes worth weighting more heavily in graph algorithms.

### 5.6 Category × City Heatmap
Pivots the exploded business table to get average rating per `(category, city)` pair and plots it as a heatmap. Green cells indicate city-category strengths — e.g., if Philadelphia's Museums score is dark green, that city has particularly well-rated museums. These patterns are embedded as node attributes in the Neo4j knowledge graph.

---

## Section 6 — Big Data Analytics Techniques

### 6.1 Sentiment Analysis via VADER

```python
@pandas_udf(FloatType())
def vader_compound(texts: pd.Series) -> pd.Series:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return texts.fillna('').apply(
        lambda t: float(sia.polarity_scores(t[:512])['compound'])
    )
```

**What this is and why:**  
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analysis tool built for social media text. It returns a `compound` score in the range [-1, +1] where +1 is maximally positive and -1 is maximally negative.

We implement it as a **Pandas UDF** (User Defined Function). A Pandas UDF is different from a regular Python UDF — instead of processing one row at a time, Spark sends batches of rows as a Pandas Series using Apache Arrow for fast serialisation. This makes it significantly faster on large datasets.

The UDF truncates text to 512 characters (`t[:512]`) to bound execution time per review — most of the sentiment signal is in the first few sentences anyway.

We apply this to a **2% random sample** of reviews because VADER is a CPU-intensive Python operation and Colab's free tier has limited resources. On a paid cluster we would process all 6 million reviews.

```python
review_sentiment = (
    review_sample
    .withColumn('sentiment_score', vader_compound(F.col('text')))
    .withColumn('sentiment_label',
        F.when(F.col('sentiment_score') >= 0.05,  'positive')
         .when(F.col('sentiment_score') <= -0.05, 'negative')
         .otherwise('neutral')
    )
    .cache()
)
```

After computing scores, we categorise them: compound ≥ 0.05 = positive, ≤ -0.05 = negative, in between = neutral. The `.cache()` call stores the result in Spark memory so the expensive UDF computation is not repeated.

```python
biz_final = (
    biz_enriched
    .join(biz_sentiment, on='business_id', how='left')
    .fillna({'avg_sentiment': 0.0})
    .withColumn('norm_sentiment', (F.col('avg_sentiment') + 1.0) / 2.0)
    .withColumn('quality_score',
        F.round(
            0.5 * F.col('norm_stars') +
            0.3 * F.col('norm_log_rev') +
            0.2 * F.col('norm_sentiment'),
            4
        )
    )
)
```

Sentiment scores are averaged per business, left-joined onto `biz_enriched`, and normalised from [-1,+1] to [0,1] by computing `(score + 1) / 2`. The final **quality score formula** is:

```
quality_score = 0.5 × norm_stars + 0.3 × norm_log_rev + 0.2 × norm_sentiment
```

The weights reflect that star rating is the strongest signal (50%), review volume provides credibility (30%), and sentiment provides a manipulation-resistant cross-check (20%).

---

### 6.2 Temporal Trend Analysis

```python
review_biz = (
    review_clean
    .join(biz_enriched.select('business_id','city','categories','is_tourism_relevant'),
          on='business_id', how='inner')
    .filter(F.col('is_tourism_relevant') == True)
    .filter(F.col('city').isin(TARGET_CITIES))
    .filter(F.col('year') >= 2016)
)

monthly_city = to_pandas(
    review_biz.groupBy('city','year','month').count().orderBy('city','year','month')
)
```

**What this is doing and why:**  
We join reviews with business data to get the city and category for each review. Then we group by city, year, and month to count how many tourism reviews each city received each month. This produces two visualisations:

1. A time-series line chart showing monthly review volume per city from 2016 onwards — reveals which cities are growing and where COVID had the biggest impact
2. A seasonality chart using only the 2016–2019 pre-COVID baseline — reveals annual patterns. Nashville peaks in spring and autumn (music festivals); Tampa peaks in winter (snowbird tourism). These seasonal patterns are stored as metadata on city nodes in Neo4j and can be used to make time-aware recommendations.

---

### 6.3 K-Means Business Clustering

```python
assembler = VectorAssembler(
    inputCols=['norm_stars','norm_log_rev','norm_sentiment'],
    outputCol='features'
)
cluster_df = assembler.transform(cluster_input)

kmeans   = KMeans(k=4, seed=42, featuresCol='features', predictionCol='cluster')
km_model = kmeans.fit(cluster_df)
clustered = km_model.transform(cluster_df)
```

**What this is and why:**  
This is the Spark MLlib (machine learning library) component of the notebook. K-Means is an unsupervised clustering algorithm that groups businesses into k=4 clusters based on their normalised star rating, review volume, and sentiment score.

`VectorAssembler` is a Spark MLlib transformer that takes multiple numeric columns and combines them into a single vector column called `features`. This is required by all Spark MLlib algorithms — they expect a single feature vector as input.

K-Means works by initialising 4 cluster centres randomly (controlled by `seed=42` for reproducibility), then iteratively assigning each business to its nearest centre and recomputing the centres until convergence.

```python
evaluator  = ClusteringEvaluator(featuresCol='features', predictionCol='cluster')
silhouette = evaluator.evaluate(clustered)
```

The **silhouette score** measures cluster quality: how similar a business is to its own cluster versus other clusters. It ranges from -1 to +1, where higher is better.

```python
centres = pd.DataFrame(km_model.clusterCenters(), ...)
centres['composite'] = 0.5*centres['norm_stars'] + 0.3*centres['norm_log_rev'] + 0.2*centres['norm_sentiment']
sorted_idx = centres['composite'].argsort().values
CLUSTER_LABELS = {
    int(sorted_idx[3]): 'Popular & Excellent',
    int(sorted_idx[2]): 'Hidden Gem',
    int(sorted_idx[1]): 'Average',
    int(sorted_idx[0]): 'Low Quality'
}
```

We assign human-readable labels by ranking the cluster centres by their composite score. The cluster with the highest composite becomes "Popular & Excellent" and the one with the lowest becomes "Low Quality". The **"Hidden Gem"** cluster is particularly valuable — it has high stars but low review volume, meaning these are places that locals love but tourists rarely discover. Surfacing these is TripGraph's core value-add over simple popularity ranking.

---

### 6.4 Co-Visit Pattern Analysis

```python
user_categories = (
    review_clean
    .join(biz_exploded.select('business_id','category')
          .filter(F.col('category').isin(TOURISM_CATEGORIES)),
          on='business_id', how='inner')
    .select('user_id','category')
    .distinct()
)

cat_pairs = (
    user_categories.alias('a')
    .join(user_categories.alias('b'), on='user_id')
    .filter(F.col('a.category') < F.col('b.category'))
    .groupBy(F.col('a.category').alias('cat_a'), F.col('b.category').alias('cat_b'))
    .count()
    .orderBy(F.desc('count'))
)
```

**What this is doing and why:**  
We want to discover which pairs of tourism categories are most frequently visited by the same user — this reveals natural travel behaviour patterns. 

The self-join (`user_categories.alias('a').join(user_categories.alias('b'), on='user_id')`) creates all pairs of categories for each user. The condition `a.category < b.category` avoids counting `(Museums, Restaurants)` and `(Restaurants, Museums)` as separate pairs — we only count each pair once (alphabetical ordering enforces this). Then we count how many users visited each pair.

The result shows that Restaurants + Coffee & Tea is the most common pair, followed by Museums + Arts & Entertainment, Parks + Food, and so on. These patterns directly inform the FP-Growth association rules in Notebook 03 and the day-grouping logic in the itinerary builder.

---

## Section 7 — Results & Export

### Top-ranked businesses per city

```python
top_per_city = to_pandas(
    biz_final
    .filter(F.col('is_tourism_relevant') == True)
    .filter(F.col('city').isin(TARGET_CITIES))
    .filter(F.col('review_count') >= 20)
    .withColumn('rank',
        F.rank().over(Window.partitionBy('city').orderBy(F.desc('quality_score')))
    )
    .filter(F.col('rank') <= 5)
)
```

**What this is doing and why:**  
This uses another window function — `F.rank().over(Window.partitionBy('city').orderBy(...))` assigns a rank to each business within its city, ordered by quality score. Then filtering to `rank <= 5` gives us the top 5 businesses per city without needing a separate query per city.

### Exporting processed data

```python
biz_final.select(...).write.mode('overwrite').parquet(f'{OUT_DIR}/businesses_clean')
review_clean.select(...).write.mode('overwrite').parquet(f'{OUT_DIR}/reviews_clean')
user_clean.filter(...).write.mode('overwrite').parquet(f'{OUT_DIR}/users_clean')
biz_exploded.select(...).write.mode('overwrite').parquet(f'{OUT_DIR}/biz_categories')
```

**What this is doing and why:**  
We save four datasets to Google Drive as **Parquet** files. Parquet is a columnar binary format that is much smaller than CSV or JSON and significantly faster to read back with Spark — it preserves data types and supports predicate pushdown (Spark can skip reading entire column groups it does not need). These files are the interface between Notebook 01 and Notebooks 02/03.

---

## Key Things to Be Able to Explain in a Viva

**"Why use Spark instead of Pandas?"**  
The Yelp dataset has 6 million reviews. Pandas loads everything into a single machine's RAM. Spark distributes the data across multiple partitions and processes them in parallel — it can handle data that does not fit in RAM using disk spill. For a dataset this size, Spark is the appropriate tool.

**"What is lazy evaluation in Spark?"**  
When you call `.withColumn()`, `.filter()`, or `.join()`, Spark does not actually execute anything — it just builds up a logical plan. Execution only happens when you call an action like `.count()`, `.show()`, or `.write()`. This allows Spark's optimizer (Catalyst) to reorder and combine operations before running them, which is more efficient than executing each step immediately.

**"Why a composite quality score instead of just using star ratings?"**  
A business with 5 stars from 3 reviews is not more trustworthy than one with 4.2 stars from 2,000 reviews. Star ratings also cluster at 3.5–4.5 due to Yelp's platform bias, making them a poor discriminating signal. Combining normalised stars (50%), log review count (30%), and sentiment (20%) gives a more robust ranking signal.

**"Why log-transform the review count?"**  
Review counts are heavily right-skewed — a power-law distribution where a few places have vastly more reviews than the rest. The log transform compresses this so that the difference between 100 and 1,000 reviews (10x) is treated more similarly to the difference between 1,000 and 10,000 (also 10x), rather than the raw counts making the outliers dominate the scoring.

**"Why normalise per city rather than globally?"**  
If we normalised globally, cities with higher average ratings would have their businesses inflate in the score relative to cities with lower averages. By normalising within each city separately, we compare each business fairly against the local competition rather than the national average.
