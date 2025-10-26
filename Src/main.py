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

# Phase 1: small datasets
subset_list = create_subsets(small=5, medium=10, large=20)
print("Created the subset files based on marketing_campaign_converted.csv")

queries_num = [50, 200]
conditions = [1,3]
t_values = [1,2,4]

spark = SparkSession.builder.appName("Phase1").getOrCreate()
for dataset in subset_list:
    for q in queries_num:
        for c in conditions:
            # Generate queries for every dataset and save to text file
            queries = generate_queries_weighted(
                spark=spark,
                csv_file_path=dataset,
                num_queries=q,
                max_conditions=c
            )

            for T in t_values:
                base_name = os.path.basename(dataset).split('.')[0]
                method1(spark, dataset, T, f"../Results/m1_{base_name}_q{q}_c{c}_T{T}.csv", load_queries(queries))
                #call other methods 
spark.stop()
# Phase 2: large datasets

