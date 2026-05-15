from neo4j import GraphDatabase

URI      = "neo4j+s://040e4230.databases.neo4j.io"
USER     = "040e4230"
PASSWORD = "o2ZlvfOSZr0ZP5Qb0sV6M21QZo29JzTJRMK4bXbBSEI"

print(f"Connecting to {URI} ...")

try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("Connection OK\n")

    with driver.session() as session:
        # Node counts
        for label in ["Business", "User", "Category", "City"]:
            count = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  {label:<12}: {count:>8,} nodes")

        print()

        # Sample business
        row = session.run("""
            MATCH (b:Business)-[:LOCATED_IN]->(ci:City)
            RETURN b.name AS name, b.city AS city, b.stars AS stars, b.quality_score AS quality
            ORDER BY b.quality_score DESC LIMIT 1
        """).single()
        if row:
            print(f"  Top business : {row['name']} ({row['city']}) {row['stars']}★")

    driver.close()
    print("\nAll checks passed — credentials are correct.")

except Exception as e:
    print(f"\nFAILED: {e}")