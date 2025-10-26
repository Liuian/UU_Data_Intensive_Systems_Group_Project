import os
from data_generator import create_subsets
from query_generator import generate_queries_weighted
from method1 import method1
from pyspark.sql import SparkSession


# Load queries from text file
def load_queries(queries_file):
    with open(queries_file, 'r') as file:
        queries = [line.strip() for line in file if line.strip()]
    print(f"Loaded {len(queries)} queries from {queries_file}")
    return queries

# Create subsets if the main dataset exists
file_path = "../Data/marketing_campaign_converted.csv"
if os.path.exists(file_path):
    print(f"File {file_path} exists, proceeding to create subsets.")
subset_list = create_subsets()
print("Created the subset files based on marketing_campaign_converted.csv")

# Generate queries and save to text file
for dataset in subset_list:
    spark_gen = SparkSession.builder.appName("QueryGenerator").getOrCreate()
    queries = generate_queries_weighted(
        spark=spark_gen,
        csv_file_path=dataset,
        num_queries=100,
        max_conditions=3
    )
    spark_gen.stop()

"""
Method 1 Execution: Dataset file, T value, Output file, Queries
"""
method1("../Data/tmp_small.csv", 30, "../Results/tmp_m1_small.csv", load_queries("../Data/tmp_small_queries.txt"))
method1("../Data/tmp_medium.csv", 100, "../Results/tmp_m1_medium.csv", load_queries("../Data/tmp_medium_queries.txt"))
method1("../Data/tmp_large.csv", 500, "../Results/tmp_m1_large.csv", load_queries("../Data/tmp_large_queries.txt"))

