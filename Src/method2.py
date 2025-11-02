from time import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import DoubleType
import sys
import math
from method1 import pop_method1
from utils import compute_query_coverage, compute_diversity
import pandas as pd
import time


# ------------------ Helper: Tuple Similarity ------------------ #
def tuple_similarity(row1, row2):
    """
    Compute similarity between two tuples based on normalized L1 distance.
    Similarity = 1 - (sum(|a_i - b_i|) / len(row))
    """
    vals1 = [v for v in row1 if isinstance(v, (int, float))]
    vals2 = [v for v in row2 if isinstance(v, (int, float))]
    if len(vals1) != len(vals2) or len(vals1) == 0:
        return 0.0
    diff = sum(abs(a - b) for a, b in zip(vals1, vals2))
    max_diff = len(vals1)
    return 1 - (diff / max_diff)


# ------------------ Method 2 Implementation ------------------ #
def method2(input_file, T, output_file, queries):
    start_time = time.time()
    spark = SparkSession.builder.appName("Method2").getOrCreate()
    print(f"\n Starting Method 2 on {input_file}\n")

    # --- Step 1: Load Dataset ---
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    print(f"Loaded {df.count()} rows and {len(df.columns)} columns.")

    # Add initial popularity column
    df = df.withColumn("popularity", lit(0))

    # --- Step 2: Compute popularity like Method 1 ---
    for query in queries:
        try:
            condition = None
            for part in query.split(" AND "):
                col_name, value = part.split(" = ")
                col_name = col_name.strip()
                value = int(value.strip())
                condition = (col(col_name) == value) if condition is None else (condition & (col(col_name) == value))

            if condition is not None:
                df = df.withColumn(
                    "popularity",
                    when(condition, col("popularity") + 1).otherwise(col("popularity"))
                )
        except Exception as e:
            print(f" Skipping invalid query '{query}': {e}")
            continue

    print("\n📊 Popularity summary:")
    df.groupBy("popularity").count().show()

    # --- Step 3: Collect data to driver for similarity calculation ---
    data = df.collect()
    num_rows = len(data)
    print(f" Computing pairwise similarities for {num_rows} tuples...")

    sim_matrix = [[0.0 for _ in range(num_rows)] for _ in range(num_rows)]
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            s = tuple_similarity(data[i], data[j])
            sim_matrix[i][j] = sim_matrix[j][i] = s

    # --- Step 4: Compute importance for each tuple ---
    importance_values = []
    for i in range(num_rows):
        pop = data[i]["total_hits"]
        if pop == 0:
            importance_values.append(0.0)
            continue
        avg_diff = sum((1 - sim_matrix[i][k]) for k in range(num_rows) if k != i) / (num_rows - 1)
        imp = pop * avg_diff
        importance_values.append(float(imp))

    # --- Step 5: Merge importance values back into DataFrame ---
    pdf = df.toPandas()
    pdf["importance"] = importance_values
    df_final = spark.createDataFrame(pdf)

    # --- Step 6: Select top T tuples ---
    top_tuples = df_final.orderBy(col("importance").desc()).limit(T)
    top_pdf = top_tuples.toPandas()

    print(f"\n Top {T} tuples by importance:")
    print(top_pdf.to_string(index=False))

    # --- Step 7: Save to CSV ---
    top_tuples.drop("popularity", "importance").write.csv(output_file, header=True, mode="overwrite")

    runtime = time.time() - start_time
    print(f"\n Method 2 completed successfully — results saved to {output_file}")
    print(f"⏱ Runtime: {runtime:.2f} seconds\n")

    # --- Step 8: Compute placeholder metrics ---
    imp_R = top_pdf["importance"].mean() if not top_pdf.empty else 0
    diversity = top_pdf["importance"].std() if not top_pdf.empty else 0
    query_coverage = df.filter(col("popularity") > 0).count() / df.count()

    spark.stop()

    # --- Step 9: Return metrics ---
    return {
        'imp_R': imp_R,
        'diversity': diversity,
        'runtime': runtime,
        'query_coverage': query_coverage
    }
