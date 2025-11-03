from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
import numpy as np
import pandas as pd
import time
from utils import parse_query_to_condition, compute_diversity_from_avg_pairwise_cosine


# ------------------ Helper: Cosine Similarity ------------------ #
def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1, dtype=float), np.array(vec2, dtype=float)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ------------------ Method 2 Implementation ------------------ #
def method2(input_file, T, output_file, queries):
    start_time = time.time()
    spark = SparkSession.builder.appName("Method2").getOrCreate()
    print(f"\n Starting Method 2 on {input_file}\n")

    # --- Step 1: Load Dataset ---
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    print(f" Loaded {df.count()} rows and {len(df.columns)} columns.")

    # Add initial popularity column
    df = df.withColumn("popularity", lit(0))

    # --- Step 2: Compute Popularity Using Queries ---
    for query in queries:
        try:
            condition = parse_query_to_condition(query)
            if condition is not None:
                df = df.withColumn(
                    "popularity",
                    when(condition, col("popularity") + 1).otherwise(col("popularity"))
                )
        except Exception as e:
            print(f" Skipping invalid query '{query}': {e}")
            continue

    print("\n Popularity summary:")
    df.groupBy("popularity").count().orderBy("popularity").show()

    # --- Step 3: Collect Data to Driver ---
    pdf = df.toPandas()
    if pdf.empty:
        print(" Empty DataFrame — aborting.")
        spark.stop()
        return None

    data_cols = [c for c in pdf.columns if c not in ["popularity"]]
    vectors = [np.array([float(x) if x is not None else 0.0 for x in row[data_cols]]) for _, row in pdf.iterrows()]
    popularities = pdf["popularity"].to_numpy()

    n = len(vectors)
    print(f" Computing similarities among {n} tuples...")

    # --- Step 4: Compute importance for each tuple ---
    importances = []
    for i in range(n):
        if popularities[i] == 0:
            importances.append(0.0)
            continue
        sims = [cosine_similarity(vectors[i], vectors[j]) for j in range(n) if j != i]
        avg_diff = np.mean([1 - s for s in sims]) if sims else 0.0
        imp_t = float(popularities[i]) * avg_diff
        importances.append(imp_t)

    pdf["importance"] = importances

    # --- Step 5: Compute normalized set importance imp(R′) ---
    pdf_sorted = pdf.sort_values(by="importance", ascending=False)
    top_T = pdf_sorted.head(int(T))
    imp_R = top_T["importance"].sum() / max(1, pdf["importance"].sum())

    # --- Step 6: Compute diversity for selected set ---
    top_vectors = [np.array([float(x) for x in row[data_cols]]) for _, row in top_T.iterrows()]
    diversity = compute_diversity_from_avg_pairwise_cosine(top_vectors)

    # --- Step 7: Compute query coverage ---
    query_coverage = (pdf[pdf["popularity"] > 0].shape[0]) / max(1, pdf.shape[0])

    # --- Step 8: Save top T tuples ---
    top_T.drop(columns=["popularity", "importance"]).to_csv(output_file, index=False)
    runtime = time.time() - start_time

    print(f"\n Top {T} tuples selected by importance.")
    print(f" imp(R′) = {imp_R:.4f}, diversity = {diversity:.4f if diversity else 0.0}, coverage = {query_coverage:.2f}")
    print(f"⏱ Runtime: {runtime:.2f} sec — results saved to {output_file}\n")

    spark.stop()

    # --- Step 9: Return metrics ---
    return {
        'imp_R': imp_R,
        'diversity': diversity,
        'runtime': runtime,
        'query_coverage': query_coverage
    }
