#%%
import time
import math
import numpy as np
from itertools import combinations
from pyspark.sql import functions as F
from utils import compute_diversity_from_avg_pairwise_cosine, parse_query_to_condition

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

def main(input_csv, T, output_csv, spark, queries):
    """
    Compute exact Shapley and write top-T. If a SparkSession is provided, use it and do NOT stop it.
    If no SparkSession is provided, create one and stop it at the end.
    """
    start_time = time.time()

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

    end_time = time.time()
    runtime = end_time - start_time

    # 7. Compute query coverage: fraction of provided queries that match at least one row in top_tuples_df
    query_coverage = None
    try:
        if queries and len(queries) > 0:
            cov_count = 0
            for q in queries:
                try:
                    cond = parse_query_to_condition(q)
                    if cond is None:
                        continue
                    if top_tuples_df.filter(cond).limit(1).count() > 0:
                        cov_count += 1
                except Exception:
                    # skip queries that fail to parse or evaluate
                    continue
            query_coverage = cov_count / max(1, len(queries))
    except Exception as e:
        print(f"Failed to compute query coverage in method3: {e}")
        query_coverage = None

    # 8. Compute imp_R, diversity for the selected top-T set
    try:
        collected_top = top_tuples_df.collect()
        # build vecs list (rows -> numpy vectors)
        vecs = []
        if collected_top:
            for r in collected_top:
                try:
                    vals = [float(r[c]) if r[c] is not None else 0.0 for c in data_col_names]
                    vecs.append(np.array(vals, dtype=float))
                except Exception:
                    continue
        # compute imp_R
        imp_R = imp_of_set(vecs)
        # compute diversity from avg_cos_sim
        diversity = compute_diversity_from_avg_pairwise_cosine(vecs)
    except Exception as e:
        print(f"Failed to compute metrics in method3: {e}")
        imp_R = None
        diversity = None

    # Return metrics to caller
    return {
        'imp_R': imp_R,
        'diversity': diversity,
        'runtime': runtime,
        'query_coverage': query_coverage
    }