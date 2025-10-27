import os
from data_generator import create_subsets
from query_generator import generate_queries_weighted
from method1 import method1
from method2 import method2
from method3 import main
from pyspark.sql import SparkSession


import os
import time
import glob
import random
import pandas as pd
import numpy as np
from data_generator import create_subsets
from query_generator import generate_queries_weighted
from method1 import method1, parse_query_to_condition
import method2
import method3 as method3_module
from pyspark.sql import SparkSession


# Load queries from text file
def load_queries(queries_file):
    with open(queries_file, 'r') as file:
        queries = [line.strip() for line in file if line.strip()]
    print(f"Loaded {len(queries)} queries from {queries_file}")
    return queries


def experiments():
    # Create subsets if the main dataset exists
    # Use full dataset path provided by user
    file_path = "./Data/marketing_campaign_cleaned.csv"
    if not os.path.exists(file_path):
        print(f"Error: main dataset {file_path} not found. Aborting experiments.")
        return

    # Define experiment parameter grid
    # queries_num_list = [50, 200]
    # conditions_list = [1, 3]
    # # (dataset_size, T) pairs
    # dataset_size_t_values = [(5, 1), (10, 2), (20, 4)]
    queries_num_list = [50]
    conditions_list = [1]
    # (dataset_size, T) pairs
    dataset_size_t_values = [(5, 1)]

    # Create subsets matching requested sizes (pass the actual dataset path so
    # create_subsets doesn't use its internal default filename)
    subset_list = create_subsets(small=5, medium=10, large=20, input_file=file_path)
    if not subset_list:
        print("No subset files created. Aborting.")
        return

    # Prepare results TSV
    results_path = "./Results/experiments_results.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_columns = [
        "method", "dataset_size", "T", "queries_num", "conditions", "seed",
        "runtime_sec", "imp_R", "avg_cosine_sim", "diversity", "query_coverage"
    ]
    results_df = pd.DataFrame(columns=results_columns)

    # Limit local parallelism to avoid creating too many native threads (pthread_create EAGAIN)
    spark = (
        SparkSession.builder
        .master("local[4]")
        .appName("ExperimentsRunner")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .getOrCreate()
    )

    # Iterate parameter grid
    for dataset in subset_list:
        # determine dataset size (number of rows)
        df_full = pd.read_csv(dataset)
        n_rows = len(df_full)
        # find matching T for this dataset size
        matching = [T for (s, T) in dataset_size_t_values if s == n_rows]
        if not matching:
            print(f"Skipping dataset {dataset}: size {n_rows} not in requested sizes")
            continue
        T_for_size = matching[0]

        for qnum in queries_num_list:
            for cond in conditions_list:
                # run 5 different random query sets
                for rep in range(5):
                    seed = random.randint(0, 2**31 - 1)
                    random.seed(seed)

                    # Generate queries (writes a file and returns path)
                    queries_file = generate_queries_weighted(
                        spark=spark,
                        csv_file_path=dataset,
                        num_queries=qnum,
                        max_conditions=cond
                    )
                    queries = load_queries(queries_file)

                    # Compute query coverage: fraction of queries matching any row in dataset
                    coverage_count = 0
                    # Use Spark DF for condition evaluation via method1 parser
                    sdf = spark.read.csv(dataset, header=True, inferSchema=True)
                    for q in queries:
                        try:
                            cond_expr = parse_query_to_condition(q)
                            if cond_expr is not None and sdf.filter(cond_expr).limit(1).count() > 0:
                                coverage_count += 1
                        except Exception:
                            continue
                    query_coverage = coverage_count / max(1, len(queries))

                    # For each method, run and measure
                    # methods_to_run = ["1", "2", "3"]
                    methods_to_run = ["3"]
                    for method_name in methods_to_run:
                        base_name = os.path.basename(dataset).split('.')[0]
                        out_dir = f"./Results/m{method_name}_dsize{base_name}_q{qnum}_c{cond}_rep{rep}_seed{seed}"
                        # ensure output dir path
                        os.makedirs(out_dir, exist_ok=True)

                        # start_time = time.time()
                        try:
                            retval = None
                            if method_name == "1":    # method1
                                retval = method1(spark, dataset, T_for_size, out_dir, queries)
                            elif method_name == "2":  # method2
                                retval = method2.method2(dataset, T_for_size, out_dir, queries)
                            else:   # method3
                                retval = method3_module.main(dataset, T_for_size, out_dir, spark, queries)
                        except Exception as e:
                            # runtime = time.time() - start_time
                            print(f"Error running {method_name} on {dataset}: {e}")
                            # record failed row with NaNs
                            row = {
                                "method": method_name,
                                "dataset_size": n_rows,
                                "T": T_for_size,
                                "queries_num": qnum,
                                "conditions": cond,
                                "seed": seed,
                                "runtime_sec": None,
                                "imp_R": None,
                                "avg_cosine_sim": None,
                                "diversity": None,
                                "query_coverage": query_coverage
                            }
                            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                            continue

                        # end_time = time.time()
                        # runtime = end_time - start_time

                        # Locate CSV file produced by Spark (part-*.csv) or direct CSV
                        csv_candidates = glob.glob(os.path.join(out_dir, "*.csv")) + glob.glob(os.path.join(out_dir, "part-*.csv"))
                        selected_csv = None
                        if csv_candidates:
                            selected_csv = csv_candidates[0]
                        else:
                            # try to find part-*.csv in directory (Spark writes as part-xxxx.csv inside folder)
                            parts = glob.glob(os.path.join(out_dir, "**", "part-*.csv"), recursive=True)
                            if parts:
                                selected_csv = parts[0]

                        imp_R = None
                        avg_cos_sim = None
                        diversity = None

                        # If the method returned metrics directly, prefer those
                        if isinstance(retval, dict):
                            imp_R = retval.get("imp_R")
                            avg_cos_sim = retval.get("avg_cosine_sim") or retval.get("avg_cos_sim")
                            diversity = retval.get("diversity")
                            runtime = retval.get("runtime")
                            # prefer method-provided query coverage if available
                            rv_qc = retval.get("query_coverage")
                            if rv_qc is not None:
                                query_coverage = rv_qc
                            # if method provides its own output path, prefer it
                            outp = retval.get("output_path")
                            if outp:
                                # if it's a directory (Spark output), look for part-*.csv inside
                                if os.path.isdir(outp):
                                    parts = glob.glob(os.path.join(outp, "*.csv")) + glob.glob(os.path.join(outp, "part-*.csv"))
                                    if parts:
                                        selected_csv = parts[0]
                                    else:
                                        inner = glob.glob(os.path.join(outp, "**", "part-*.csv"), recursive=True)
                                        if inner:
                                            selected_csv = inner[0]
                                elif os.path.isfile(outp):
                                    selected_csv = outp
                                else:
                                    # treat as candidate path anyway
                                    selected_csv = outp
                        else:
                            if selected_csv and os.path.exists(selected_csv):
                                try:
                                    res_df = pd.read_csv(selected_csv)
                                    # assume first column is ID, rest are numeric features
                                    if res_df.shape[0] > 0:
                                        vec_cols = res_df.columns[1:]
                                        vecs = res_df[vec_cols].to_numpy(dtype=float)
                                        # compute centroid
                                        centroid = np.mean(vecs, axis=0)
                                        sims = []
                                        for v in vecs:
                                            na = np.linalg.norm(v)
                                            nc = np.linalg.norm(centroid)
                                            sim = 0.0
                                            if na != 0 and nc != 0:
                                                sim = float(np.dot(v, centroid) / (na * nc))
                                            sims.append(sim)
                                        dissimilarities = [1.0 - s for s in sims]
                                        imp_R = float(np.mean(dissimilarities))

                                        # average pairwise cosine similarity
                                        pair_sims = []
                                        m = len(vecs)
                                        if m > 1:
                                            for i in range(m):
                                                for j in range(i + 1, m):
                                                    a = vecs[i]
                                                    b = vecs[j]
                                                    na = np.linalg.norm(a)
                                                    nb = np.linalg.norm(b)
                                                    if na == 0 or nb == 0:
                                                        s = 0.0
                                                    else:
                                                        s = float(np.dot(a, b) / (na * nb))
                                                    pair_sims.append(s)
                                            avg_cos_sim = float(np.mean(pair_sims))
                                        else:
                                            avg_cos_sim = 1.0 if m == 1 else None
                                        diversity = 1.0 - avg_cos_sim if avg_cos_sim is not None else None
                                except Exception as e:
                                    print(f"Failed to compute metrics from {selected_csv}: {e}")

                        # Append results row
                        row = {
                            "method": method_name,
                            "dataset_size": n_rows,
                            "T": T_for_size,
                            "queries_num": qnum,
                            "conditions": cond,
                            "seed": seed,
                            "runtime_sec": runtime,
                            "imp_R": imp_R,
                            "avg_cosine_sim": avg_cos_sim,
                            "diversity": diversity,
                            "query_coverage": query_coverage
                        }
                        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

                        # persist intermediate results to disk
                        results_df.to_csv(results_path, index=False)

    spark.stop()
    print(f"Experiments finished. Results written to {results_path}")


# def testMethod1(file_path):
#     spark = SparkSession.builder.appName("TestMethod1").getOrCreate()
#     queries = generate_queries_weighted(
#                     spark=spark,
#                     csv_file_path=file_path,
#                     num_queries=20,
#                     max_conditions=3
#                 )
#     method1(spark, file_path, 100, "../Results/m1_test.csv", load_queries(queries))
#     spark.stop()


# def testMethod2(file_path):
#     # todo: implement test for method2
#     pass
    
# def testMethod3(file_path):
#     # todo: implement test for method3
#     pass

if __name__ == "__main__":
    # run experiments
    experiments()
    # testMethod1("./Data/marketing_campaign_cleaned.csv")
