"""
Recommendation engine — queries Neo4j and computes hybrid scores.
No PySpark required; all computation is done with scikit-learn + pandas.
"""

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    TOURISM_CATEGORIES, HYBRID_WEIGHTS, DEFAULT_TOP_N,
)


class TripGraphRecommender:
    """
    Loads business data from Neo4j once at startup and exposes four
    recommendation methods: content-based, graph PageRank, quality-ranked,
    and a hybrid that combines all three.
    """

    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._biz_df = None
        self._tfidf_matrix = None
        self._vectorizer = None

    # ── Neo4j helpers ─────────────────────────────────────────────────────────

    def _run(self, query: str, **params) -> list[dict]:
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, **params)]

    def close(self):
        self.driver.close()

    # ── Business cache (loaded once per Streamlit session) ────────────────────

    def load_businesses(self) -> pd.DataFrame:
        """
        Pull all Business nodes from Neo4j into a Pandas DataFrame.
        Called once; result is cached in self._biz_df.
        """
        if self._biz_df is not None:
            return self._biz_df

        rows = self._run("""
            MATCH (b:Business)
            OPTIONAL MATCH (b)-[:IN_CATEGORY]->(c:Category)
            WITH b, collect(c.name) AS cats
            RETURN
                b.business_id   AS business_id,
                b.name          AS name,
                b.city          AS city,
                b.state         AS state,
                b.lat           AS lat,
                b.lon           AS lon,
                b.stars         AS stars,
                b.review_count  AS review_count,
                b.price_label   AS price_label,
                b.quality_score AS quality_score,
                b.avg_sentiment AS avg_sentiment,
                b.categories    AS categories,
                b.community_id  AS community_id,
                cats            AS category_list
        """)

        df = pd.DataFrame(rows)
        df["quality_score"]  = pd.to_numeric(df["quality_score"],  errors="coerce").fillna(0)
        df["avg_sentiment"]  = pd.to_numeric(df["avg_sentiment"],  errors="coerce").fillna(0)
        df["stars"]          = pd.to_numeric(df["stars"],          errors="coerce").fillna(0)
        df["review_count"]   = pd.to_numeric(df["review_count"],   errors="coerce").fillna(0)
        df["lat"]            = pd.to_numeric(df["lat"],            errors="coerce")
        df["lon"]            = pd.to_numeric(df["lon"],            errors="coerce")

        # Build category text for TF-IDF
        def _cat_text(row):
            cats = row.get("category_list") or []
            orig = row.get("categories") or ""
            return " ".join(cats) + " " + orig

        df["cat_text"] = df.apply(_cat_text, axis=1)
        df["cat_text"] = df["cat_text"].str.lower().str.replace(r"[^a-z ]", " ", regex=True)

        self._biz_df = df
        self._fit_tfidf()
        return self._biz_df

    def _fit_tfidf(self):
        self._vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._biz_df["cat_text"].fillna(""))

    # ── Method A: Content-based filtering ────────────────────────────────────

    def content_recommend(
        self,
        city: str,
        categories: list[str],
        budget: str | None = None,
        top_n: int = DEFAULT_TOP_N,
    ) -> pd.DataFrame:
        """Rank businesses by TF-IDF cosine similarity to user preferences."""
        df = self.load_businesses()

        # Build query vector
        query_text = re.sub(r"[^a-z ]", " ", " ".join(categories).lower())
        query_vec  = self._vectorizer.transform([query_text])

        # Compute similarities across the full matrix
        sims = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        df = df.copy()
        df["content_score"] = sims

        # Filter to city + budget
        mask = df["city"] == city
        if budget:
            mask &= df["price_label"] == budget
        city_df = df[mask].copy()

        return (
            city_df
            .sort_values("content_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ── Method B: Graph Personalized PageRank ─────────────────────────────────

    def graph_recommend(
        self,
        city: str,
        categories: list[str],
        top_n: int = DEFAULT_TOP_N,
    ) -> pd.DataFrame:
        """
        Run Personalized PageRank seeded on businesses matching the user's
        preferred categories. Falls back to quality ranking if GDS is unavailable.
        """
        # Find seed businesses
        seed_rows = self._run("""
            MATCH (b:Business)-[:LOCATED_IN]->(ci:City {name: $city})
            MATCH (b)-[:IN_CATEGORY]->(c:Category)
            WHERE c.name IN $categories
            RETURN b.business_id AS business_id
            ORDER BY b.quality_score DESC
            LIMIT 10
        """, city=city, categories=categories)

        seed_ids = [r["business_id"] for r in seed_rows]

        if not seed_ids:
            # No seeds — fall back to quality ranking
            return self._quality_fallback(city, top_n)

        try:
            rows = self._run("""
                MATCH (source:Business)
                WHERE source.business_id IN $seed_ids
                WITH collect(source) AS sourceNodes
                CALL gds.pageRank.stream('tripgraph', {
                    sourceNodes: sourceNodes,
                    dampingFactor: 0.85,
                    maxIterations: 20,
                    nodeLabels: ['Business']
                })
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS b, score
                WHERE b.city = $city AND score > 0
                RETURN
                    b.business_id   AS business_id,
                    score           AS pagerank_score
                ORDER BY score DESC
                LIMIT $top_n
            """, seed_ids=seed_ids, city=city, top_n=top_n)

            if rows:
                ppr_df = pd.DataFrame(rows)
                df = self.load_businesses()
                merged = df[df["city"] == city].merge(ppr_df, on="business_id", how="left")
                merged["pagerank_score"] = merged["pagerank_score"].fillna(0.0)
                return merged.sort_values("pagerank_score", ascending=False).head(top_n).reset_index(drop=True)

        except Exception:
            pass

        # GDS not available — use category-match quality fallback
        return self._quality_fallback(city, top_n, categories)

    def _quality_fallback(self, city, top_n, categories=None):
        df = self.load_businesses()
        city_df = df[df["city"] == city].copy()
        if categories:
            pattern = "|".join(re.escape(c) for c in categories)
            city_df = city_df[city_df["categories"].str.contains(pattern, case=False, na=False)]
        city_df["pagerank_score"] = city_df["quality_score"]
        return city_df.sort_values("pagerank_score", ascending=False).head(top_n).reset_index(drop=True)

    # ── Method C: Hybrid scoring ──────────────────────────────────────────────

    def hybrid_recommend(
        self,
        city: str,
        categories: list[str],
        budget: str | None = None,
        top_n: int = DEFAULT_TOP_N,
    ) -> pd.DataFrame:
        """
        Combine content-based, graph PageRank, and quality scores.
        Weights from config.HYBRID_WEIGHTS.
        """
        content_df = self.content_recommend(city, categories, budget=budget, top_n=top_n * 2)
        graph_df   = self.graph_recommend(city, categories, top_n=top_n * 2)

        # Merge on business_id
        merged = content_df.merge(
            graph_df[["business_id", "pagerank_score"]].drop_duplicates("business_id"),
            on="business_id",
            how="left",
        )
        merged["pagerank_score"] = merged["pagerank_score"].fillna(0.0)

        def _minmax(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-9)

        merged["norm_content"]  = _minmax(merged["content_score"])
        merged["norm_pagerank"] = _minmax(merged["pagerank_score"])
        merged["norm_quality"]  = _minmax(merged["quality_score"])

        w = HYBRID_WEIGHTS
        merged["hybrid_score"] = (
            w["graph"]   * merged["norm_pagerank"] +
            w["content"] * merged["norm_content"]  +
            w["quality"] * merged["norm_quality"]
        )

        return (
            merged
            .sort_values("hybrid_score", ascending=False)
            .drop_duplicates("business_id")
            .head(top_n)
            .reset_index(drop=True)
        )

    # ── Why this place? explanation ───────────────────────────────────────────

    def explain(self, row: pd.Series, user_categories: list[str]) -> str:
        """Return a one-sentence explanation for why a business was recommended."""
        name    = row.get("name", "This place")
        stars   = row.get("stars", 0)
        n_rev   = int(row.get("review_count", 0))
        quality = row.get("quality_score", 0)
        price   = row.get("price_label", "")
        cats    = row.get("category_list") or []

        # Find matching categories
        matched = [c for c in user_categories if any(c.lower() in cat.lower() for cat in cats)]

        if quality > 0.75:
            tier = "one of the highest-rated spots"
        elif quality > 0.5:
            tier = "a well-regarded spot"
        else:
            tier = "a solid local pick"

        if matched:
            cat_str = " and ".join(matched[:2])
            reason  = f"matches your interest in {cat_str}"
        else:
            reason  = "fits your budget and travel style"

        return (
            f"**{name}** is {tier} in {row.get('city', 'the city')} "
            f"({stars:.1f}★, {n_rev:,} reviews, {price} budget) — {reason}."
        )

    # ── City stats ─────────────────────────────────────────────────────────────

    def city_stats(self, city: str) -> dict:
        rows = self._run("""
            MATCH (b:Business)-[:LOCATED_IN]->(ci:City {name: $city})
            RETURN
                count(b)          AS total_businesses,
                avg(b.stars)      AS avg_rating,
                sum(b.review_count) AS total_reviews
        """, city=city)
        return rows[0] if rows else {}
