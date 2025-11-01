from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, array, expr, size, desc
from utils import compute_query_coverage, compute_diversity
import time

def pop_method1(df, queries):
    """
    Select top T tuples based on total hits from queries.
    """

    # Assign QID's for every Query
    queries_with_ids = [(qid+1, q) for qid, q in enumerate(queries)]

    # Compute total hits directly without building arrays
    total_hits_expr = sum(
        when(expr(expr_str), lit(1)).otherwise(lit(0))
        for _, expr_str in queries_with_ids
    )

    df_total_hits = df.withColumn("total_hits", total_hits_expr)

    # Select top T rows by total_hits
    df_pop = df_total_hits.orderBy(desc("total_hits"))
    
    return df_pop

def method1(spark, input_file, T, output_file, queries):
    # Store original queries for coverage calculation
    original_queries = queries.copy()
    start_time = time.time()

    # Read input data
    df = spark.read.csv(input_file, header=True, inferSchema=True)

    # Get top T tuples based on popularity
    df_topT = pop_method1(df, queries).limit(T)

    end_time = time.time()
    
    # Sum of total hits in R'
    imp_R = df_topT.agg({'total_hits': 'sum'}).collect()[0][0]

    # Save to CSV
    df_topT.drop("total_hits").write.csv(output_file, header=True, mode='overwrite')
    
    # Use original queries (without IDs) for coverage calculation
    query_coverage = compute_query_coverage(spark, df_topT, original_queries)
    diversity = compute_diversity(df_topT, df.columns)

    

    return {
        'imp_R': imp_R, 
        'diversity': diversity,
        'runtime': end_time - start_time,
        'query_coverage': query_coverage
    }