#%%
import os
# Set environment / JVM flags before importing pyspark so the JVM picks them up.
# This avoids UserGroupInformation/Subject errors on newer JDKs for local runs.
os.environ.setdefault("HADOOP_USER_NAME", "sparkuser")
# Ensure the JVM option is applied early for driver and executor
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Djavax.security.auth.useSubjectCredsOnly=false")
# Also pass via PYSPARK_SUBMIT_ARGS for pyspark-shell initialization
os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS",
    "--conf spark.driver.extraJavaOptions=-Djavax.security.auth.useSubjectCredsOnly=false --conf spark.executor.extraJavaOptions=-Djavax.security.auth.useSubjectCredsOnly=false pyspark-shell",
)

import csv
import random
import numpy as np
from pyspark.sql import SparkSession, functions as F

#%% Functions
def read_csv_collect_rows(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            rows.append(row)
    return header, rows

def row_to_vector(row, cols_to_use=None):
    # convert selected columns (strings) to integer vector (dataset already integers)
    if cols_to_use is None:
        cols_to_use = range(len(row))
    # cast to int first (dataset values are integers), then to float for numeric ops
    vec = [int(row[i]) for i in cols_to_use]
    return np.array(vec, dtype=float)

def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def imp_of_set(vectors):
    # imp = average(1 - sim(t_i, centroid))
    if len(vectors) == 0:
        return 0.0
    centroid = np.mean(vectors, axis=0)
    sims = [cosine_similarity(v, centroid) for v in vectors]
    dissimilarities = [1.0 - s for s in sims]
    return float(np.mean(dissimilarities))

def approximate_shapley(vectors, ids, num_samples=100, seed=123):
    # vectors: list of numpy arrays; ids: same order identifiers
    n = len(vectors)
    acc = {ids[i]: 0.0 for i in range(n)}
    random.seed(seed)
    for s in range(num_samples):
        order = list(range(n))
        random.shuffle(order)
        prefix_vectors = []
        imp_prefix = 0.0
        # iterate in permutation order
        for idx in order:
            v = vectors[idx]
            # marginal = imp(prefix ∪ {v}) - imp(prefix)
            new_vectors = prefix_vectors + [v]
            imp_new = imp_of_set(new_vectors)
            marginal = imp_new - imp_prefix
            acc[ids[idx]] += marginal
            # update prefix
            prefix_vectors = new_vectors
            imp_prefix = imp_new
    # average over samples
    for k in acc:
        acc[k] = acc[k] / num_samples
    return acc

def main(input_csv, T, output_csv, num_samples=100):
    # 1. Start Spark Session
    # Add configuration to handle JDK 17+ incompatibility with Hadoop's UserGroupInformation
    # Set a simple HADOOP user for local runs to avoid UGI issues
    os.environ.setdefault("HADOOP_USER_NAME", "sparkuser")

    # JVM system property that avoids Subject usage in some Java versions
    java_auth_flag = "-Djavax.security.auth.useSubjectCredsOnly=false"

    # Build Spark session for local execution; include extraJavaOptions to pass JVM flag
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("method3_shapley")
        .config("spark.hadoop.fs.defaultFS", "file:/")
        .config("spark.hadoop.hadoop.security.user.provider.class", "org.apache.hadoop.security.authentication.util.KerberosName$NoOpUserProvider")
        .config("spark.driver.extraJavaOptions", java_auth_flag)
        .config("spark.executor.extraJavaOptions", java_auth_flag)
        .getOrCreate()
    )

    # 2. Use Spark to read and process data
    # Read CSV, infer schema, and use the first row as header
    df = spark.read.csv(input_csv, header=True, inferSchema=True)

    # Get ID column name and data column names
    id_col_name = df.columns[0]
    data_col_names = df.columns[1:]

    # Collect all data to the Driver for Shapley computation
    # Note: if the dataset is very large this step may cause Driver OOM.
    # In that case, a distributed Shapley implementation is needed.
    print("Read data with Spark and collect to Driver...")
    collected_rows = df.select(id_col_name, *data_col_names).collect()
    print(f"Collected {len(collected_rows)} rows.")

    # 3. Convert Spark Rows to Python/Numpy objects
    # This is to be compatible with the existing approximate_shapley function
    ids = [str(row[id_col_name]) for row in collected_rows]
    vectors = [np.array(row[1:], dtype=float) for row in collected_rows]

    # 4. Run Shapley value estimation (runs on the Driver)
    print(f"Starting Shapley estimation with {num_samples} samples...")
    scores = approximate_shapley(vectors, ids, num_samples=int(num_samples))

    # 5. Select top T items by score
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Convert IDs back to original data type (assumed int) for filtering
    top_ids = [int(iid) for iid, sc in sorted_ids[:int(T)]]



    # 6. Use Spark to filter and write results
    # Create a DataFrame containing only the top IDs for join/filter
    top_ids_df = spark.createDataFrame([(i,) for i in top_ids], [id_col_name])

    # Filter rows in the original DataFrame matching the top IDs
    top_tuples_df = df.join(F.broadcast(top_ids_df), id_col_name, "inner")

    # Write result as a single CSV file
    top_tuples_df.coalesce(1).write.csv(output_csv, header=True, mode="overwrite")

    # 7. Stop Spark Session
    spark.stop()
    print(f"Done. Wrote top {T} items to directory {output_csv} using Spark")

#%% --- 1. Parameters ---
# Please set input/output file paths and parameters here
INPUT_CSV_PATH = "/Users/peggy/Documents/uu_master_data_science/uu_data_intensive_systems_group_project/Data/marketing_campaign_cleaned.csv"
OUTPUT_CSV_PATH = "/Users/peggy/Documents/uu_master_data_science/uu_data_intensive_systems_group_project/Data/method3_output.csv"
T_VALUE = 20  # 要選取的頂部資料筆數
NUM_SAMPLES = 10  # Shapley 估計的取樣次數

#%% --- 2. Run main script ---
print("Starting Shapley value computation...")
main(INPUT_CSV_PATH, T_VALUE, OUTPUT_CSV_PATH, NUM_SAMPLES)
print("Program finished.")
