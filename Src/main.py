import os
import random
import pandas as pd
from pyspark.sql import SparkSession
from method1 import method1
from method2 import method2
import method3 as method3_module
from utils import create_subsets, generate_queries_weighted
from config import *

# Load queries from text file
def load_queries(queries_file):
    with open(queries_file, 'r') as file:
        queries = [line.strip() for line in file if line.strip()]
    print(f"Loaded {len(queries)} queries from {queries_file}")
    return queries

def experiments(RESULTS_PATH, REP_RANGE, QUERIES_NUM_LIST, CONDITIONS_LIST, DATASET_SIZE_T_VALUES, METHODS_TO_RUN, SUBSET_SIZES):
    # Create subsets if the main dataset exists
    # Use full dataset path provided by user
    if not os.path.exists(FILE_PATH):
        print(f"Error: main dataset {FILE_PATH} not found. Aborting experiments.")
        return

    # create_subsets
    subset_list = create_subsets(small=SUBSET_SIZES[0], medium=SUBSET_SIZES[1], large=SUBSET_SIZES[2], input_file=FILE_PATH)
    if not subset_list:
        print("No subset files created. Aborting.")
        return

    # Prepare results TSV
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
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
        matching = [T for (s, T) in DATASET_SIZE_T_VALUES if s == n_rows]
        if not matching:
            print(f"Skipping dataset {dataset}: size {n_rows} not in requested sizes")
            continue
        T_for_size = matching[0]

        for qnum in QUERIES_NUM_LIST:
            for cond in CONDITIONS_LIST:
                # run 5 different random query sets
                for rep in range(REP_RANGE):
                    seed = random.randint(0, 2**31 - 1)
                    random.seed(seed)

                    # Generate queries (writes a file and returns path)
                    queries_file = generate_queries_weighted(
                        spark=spark,
                        csv_file_path=dataset,
                        num_queries=qnum,
                        max_conditions=cond,
                        seed=seed
                    )
                    queries = load_queries(queries_file)

                    # For each method, run and measure
                    for method_name in METHODS_TO_RUN:
                        base_name = os.path.basename(dataset).split('.')[0]
                        OUT_DIR = f"./Results/m{method_name}_dsize{base_name}_q{qnum}_c{cond}_rep{rep}_seed{seed}"
                        os.makedirs(OUT_DIR, exist_ok=True) # ensure output dir path

                        try:
                            retval = None
                            runtime = None
                            if method_name == "1":    # method1
                                retval = method1(spark, dataset, T_for_size, OUT_DIR, queries)
                            elif method_name == "2":  # method2
                                retval = method2.method2(dataset, T_for_size, OUT_DIR, queries)
                            else:   # method3
                                retval = method3_module.main(dataset, T_for_size, OUT_DIR, spark, queries)
                        except Exception as e:
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

                        # If the method returned metrics directly, prefer those
                        if isinstance(retval, dict):
                            imp_R = retval.get("imp_R")
                            avg_cos_sim = retval.get("avg_cosine_sim") or retval.get("avg_cos_sim")
                            diversity = retval.get("diversity")
                            runtime = retval.get("runtime")
                            query_coverage = retval.get("query_coverage")
                        else:
                            # If the method didn't return metrics, leave metrics as None.
                            pass

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

                        results_df.to_csv(RESULTS_PATH, index=False)

    spark.stop()
    print(f"Experiments finished. Results written to {RESULTS_PATH}")

if __name__ == "__main__":
    experiments(RESULTS_PATH, REP_RANGE, QUERIES_NUM_LIST, CONDITIONS_LIST, DATASET_SIZE_T_VALUES, METHODS_TO_RUN, SUBSET_SIZES)
    # experiments(EXP1_RESULTS_PATH, EXP1_REP_RANGE, EXP1_QUERIES_NUM_LIST, EXP1_CONDITIONS_LIST, EXP1_DATASET_SIZE_T_VALUES,EXP1_METHODS_TO_RUN, EXP1_SUBSET_SIZES)
    # experiments(EXP2_RESULTS_PATH, EXP2_REP_RANGE, EXP2_QUERIES_NUM_LIST, EXP2_CONDITIONS_LIST, EXP2_DATASET_SIZE_T_VALUES,EXP2_METHODS_TO_RUN, EXP2_SUBSET_SIZES)