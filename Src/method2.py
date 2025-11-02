from time import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import DoubleType
import sys
import math
from method1 import pop_method1
from utils import compute_query_coverage, compute_diversity

def tuple_similarity(row1, row2):
    """
    Compute similarity between two tuples based on normalized L1 distance.
    Similarity = 1 - (sum(|a_i - b_i|) / max_possible_diff)
    """
    vals1 = [v for v in row1 if isinstance(v, (int, float))]
    vals2 = [v for v in row2 if isinstance(v, (int, float))]
    if len(vals1) != len(vals2) or len(vals1) == 0:
        return 0
    diff = sum(abs(a - b) for a, b in zip(vals1, vals2))
    max_diff = len(vals1) * 1.0  # normalized scale
    return 1 - (diff / max_diff)


def method2(spark, input_file, T, output_file, queries):
    # Store original queries for coverage calculation
    original_queries = queries.copy()
    start_time = time.time()

    # Read input data
    df = spark.read.csv(input_file, header=True, inferSchema=True)

    # Compute popularity using method1 (based on total hits)
    df_with_popularity = pop_method1(df, queries)

    # Collect data to driver for similarity computations
    data = df_with_popularity.collect()
    num_rows = len(data)

    # Build similarity matrix
    sim_matrix = [[0.0 for _ in range(num_rows)] for _ in range(num_rows)]   
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            s = tuple_similarity(data[i], data[j])
            sim_matrix[i][j] = sim_matrix[j][i] = s

    # Compute importance values
    importance_values = []
    for i in range(num_rows):
        pop = data[i]["total_hits"]
        if pop == 0:
            importance_values.append(0)
            continue
        avg_diff = sum((1 - sim_matrix[i][k]) for k in range(num_rows) if k != i) / (num_rows - 1)
        imp = pop * avg_diff
        importance_values.append(imp)

    # Add importance column back to DataFrame
    df_with_importance = df_with_popularity.withColumn("importance", lit(0.0).cast(DoubleType()))
    
    for idx, imp in enumerate(importance_values):
        df_with_importance = df_with_importance.withColumn(
            "importance",
            when(
                (col(df_with_importance.columns[0]) == data[idx][0]),  # simple ID match
                lit(imp)
            ).otherwise(col("importance"))
        )

    # Select top T tuples
    top_tuples = df_with_importance.orderBy(col("importance").desc()).limit(T)
    
    # save to CSV
    top_tuples.drop("total_hits", "importance").write.csv(output_file, header=True, mode='overwrite')

    # Use original queries (without IDs) for coverage calculation
    query_coverage = compute_query_coverage(spark, top_tuples, original_queries)
    diversity = compute_diversity(top_tuples, df.columns)
    