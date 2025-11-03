# method2.py
import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, expr, size, desc
from utils import compute_query_coverage, compute_diversity

def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def pop_method2(df, queries):
    """Compute popularity of each tuple (same as method1)."""
    queries_with_ids = [(qid + 1, q) for qid, q in enumerate(queries)]
    total_hits_expr = sum(
        when(expr(expr_str), lit(1)).otherwise(lit(0))
        for _, expr_str in queries_with_ids
    )
    return df.withColumn("popularity", total_hits_expr)

def method2(input_file, T, output_file, queries):
    start_time = time.time()

    # Initialize Spark locally
    spark = SparkSession.builder.master("local[*]").appName("Method2").getOrCreate()

    # 1. Read CSV
    df = spark.read.csv(input_file, header=True, inferSchema=True)

    # 2. Compute popularity for each tuple
    df_pop = pop_method2(df, queries)
    data_cols = [c for c in df.columns]

    # 3. Collect to driver for pairwise computations
    rows = df_pop.collect()
    vectors = []
    pops = []
    for r in rows:
        try:
            vec = np.array([float(r[c]) if r[c] is not None else 0.0 for c in data_cols])
            vectors.append(vec)
            pops.append(float(r["popularity"]))
        except Exception:
            continue

    # 4. Compute pairwise similarities (symmetric matrix)
    n = len(vectors)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_similarity(vectors[i], vectors[j])
            sim[i, j] = s
            sim[j, i] = s

    # 5. Compute weighted importance: pop(t) * average(1 - sim(t, others))
    importance = []
    for i in range(n):
        if n > 1:
            avg_diff = np.mean(1 - sim[i, :])
        else:
            avg_diff = 1.0
        imp_t = pops[i] * avg_diff
        importance.append(imp_t)

    # 6. Rank tuples by importance
    idx_sorted = np.argsort(importance)[::-1]
    top_idx = idx_sorted[:T]

    # Build top-T DataFrame
    top_ids = [rows[i][0] for i in top_idx]  # assuming first col is ID
    top_ids_df = spark.createDataFrame([(i,) for i in top_ids], [df.columns[0]])
    df_topT = df.join(top_ids_df, df.columns[0], "inner")

    # 7. Compute metrics
    imp_R = np.sum(importance[top_idx]) / np.sum(importance)
    query_coverage = compute_query_coverage(spark, df_topT, queries)
    diversity = compute_diversity(df_topT, df.columns)

    runtime = time.time() - start_time

    # 8. Save top-T
    df_topT.write.csv(output_file, header=True, mode="overwrite")

    spark.stop()

    return {
        "imp_R": imp_R,
        "diversity": diversity,
        "runtime": runtime,
        "query_coverage": query_coverage
    }
