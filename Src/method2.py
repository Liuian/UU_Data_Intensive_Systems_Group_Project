import os
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, expr, lit
from utils import compute_query_coverage, compute_diversity_from_avg_pairwise_cosine


def safe_numeric_matrix(pdf):
    """Return numeric NumPy matrix & list of numeric columns."""
    numeric_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        for c in pdf.columns:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
        numeric_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        return np.zeros((len(pdf), 1)), []
    mat = pdf[numeric_cols].fillna(0).astype(float).to_numpy()
    return mat, numeric_cols


def pop_method2(spark, df, queries):
    """Compute popularity × dissimilarity importance."""
    total_hits_expr = sum(when(expr(q), lit(1)).otherwise(lit(0)) for q in queries)
    df_pop = df.withColumn("popularity", total_hits_expr)
    pdf = df_pop.toPandas()

    mat, numeric_cols = safe_numeric_matrix(pdf)
    if len(pdf) == 0:
        pdf["importance"] = 0.0
        return pdf, numeric_cols

    try:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        sim_matrix = np.dot(mat, mat.T) / (norms @ norms.T)
        sim_matrix = np.clip(sim_matrix, -1, 1)
        dissim_matrix = 1 - sim_matrix
        avg_dissim = dissim_matrix.mean(axis=1)
        pdf["importance"] = pdf["popularity"].astype(float) * avg_dissim
    except Exception as e:
        print("[Method2] Similarity failed:", e)
        pdf["importance"] = pdf["popularity"]

    return pdf, numeric_cols


def method2(input_file, T, output_dir, queries):
    """Popularity × dissimilarity weighting — safe, consistent with Method 1 & 3."""
    spark = SparkSession.builder.getOrCreate()
    start = time.time()

    # ---------- Load data ----------
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    pdf, numeric_cols = pop_method2(spark, df, queries)

    # ---------- Compute imp_R ----------
    try:
        pdf_sorted = pdf.sort_values(by="importance", ascending=False)
        topT = pdf_sorted.head(T)
        total_imp = float(pdf_sorted["importance"].sum())
        top_imp = float(topT["importance"].sum())
        imp_R = top_imp / total_imp if total_imp != 0 else 0.0
    except Exception as e:
        print("[Method2] Importance calc failed:", e)
        imp_R = 0.0
        topT = pdf.head(T)

    runtime = time.time() - start

    # ---------- Save output ----------
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, "topT.csv")
        topT.drop(columns=["importance", "popularity"], errors="ignore").to_csv(out_csv, index=False)
    except Exception as e:
        print("[Method2] Save failed:", e)

    # ---------- Diversity ----------
    try:
        if len(numeric_cols) > 0 and len(topT) > 1:
            diversity = float(compute_diversity_from_avg_pairwise_cosine(topT[numeric_cols].to_numpy()))
        else:
            diversity = 0.0
    except Exception as e:
        print("[Method2] Diversity failed:", e)
        diversity = 0.0

    # ---------- Query coverage ----------
    try:
        spark_topT = spark.createDataFrame(topT)
        query_coverage = float(compute_query_coverage(spark, spark_topT, queries))
    except Exception as e:
        print("[Method2] Coverage failed:", e)
        query_coverage = 0.0

    print(f"[Method2] Done: runtime={runtime:.4f}s imp_R={imp_R:.4f} div={diversity:.4f} cov={query_coverage:.4f}")

    return {
        "imp_R": round(imp_R, 6),
        "diversity": round(diversity, 6),
        "runtime": round(runtime, 6),
        "query_coverage": round(query_coverage, 6)
    }
