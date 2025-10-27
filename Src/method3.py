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
from itertools import combinations
import math
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

def exact_shapley(vectors, ids):
    """
    Compute exact Shapley values for each element using the importance function
    imp(R') = average_{t in R'} (1 - sim(t, centroid(R'))).

    This enumerates all subsets S of R \ {t} and uses the exact Shapley weight:
        weight = |S|! * (n - |S| - 1)! / n!

    Complexity is exponential; we enforce a safety limit on n (default 20).
    """
    n = len(vectors)
    if n == 0:
        return {}
    # Safety guard: exact enumeration is O(n * 2^n). Adjust as needed.
    if n > 20:
        raise ValueError(f"Exact Shapley is exponential; n must be <= 20. Current n={n}")

    acc = {ids[i]: 0.0 for i in range(n)}
    fact = [math.factorial(i) for i in range(n + 1)]
    denom = fact[n]
    indices = list(range(n))

    # For each element t, sum marginal contributions over all subsets S ⊆ R \ {t}
    for i in range(n):
        others = [j for j in indices if j != i]
        # iterate subset sizes
        for r in range(0, len(others) + 1):
            # iterate subsets of size r
            for S in combinations(others, r):
                S_vectors = [vectors[j] for j in S]
                imp_S = imp_of_set(S_vectors)
                imp_Si = imp_of_set(S_vectors + [vectors[i]])
                marginal = imp_Si - imp_S
                weight = (fact[len(S)] * fact[n - len(S) - 1]) / denom
                acc[ids[i]] += weight * marginal
    return acc

def main(input_csv, T, output_csv):
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

    # 4. Run exact Shapley value estimation (runs on the Driver)
    print("Starting exact Shapley estimation (exact enumeration)...")
    scores = exact_shapley(vectors, ids)

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
INPUT_CSV_PATH = "/Users/peggy/Documents/uu_master_data_science/uu_data_intensive_systems_group_project/Data/data_test_10.csv"
OUTPUT_CSV_PATH = "/Users/peggy/Documents/uu_master_data_science/uu_data_intensive_systems_group_project/Data/method3_output.csv"
T_VALUE = 4  # 要選取的頂部資料筆數

#%% --- 2. Run main script ---
print("Starting Shapley value computation...")
main(INPUT_CSV_PATH, T_VALUE, OUTPUT_CSV_PATH)
print("Program finished.")
