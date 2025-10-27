from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import DoubleType
import sys
import math

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


def method2(input_file, T, output_file, queries):
    
    spark = SparkSession.builder \
        .appName("Method2") \
        .getOrCreate()
    
   
    df = spark.read.csv(input_file, header=True, inferSchema=True)

    
    df = df.withColumn("popularity", lit(0))

    
    for query in queries:
        try:
            condition = None
            for part in query.split(" AND "):
                col_name, value = part.split(" = ")
                col_name = col_name.strip()
                value = int(value.strip())

                if condition is None:
                    condition = (col(col_name) == value)
                else:
                    condition = condition & (col(col_name) == value)

            if condition is not None:
                df = df.withColumn(
                    "popularity",
                    when(condition, col("popularity") + 1).otherwise(col("popularity"))
                )
        except:
            continue

    
    data = df.collect()
    num_rows = len(data)
    sim_matrix = [[0.0 for _ in range(num_rows)] for _ in range(num_rows)]

    
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            s = tuple_similarity(data[i], data[j])
            sim_matrix[i][j] = sim_matrix[j][i] = s

    
    importance_values = []
    for i in range(num_rows):
        pop = data[i]["popularity"]
        if pop == 0:
            importance_values.append(0)
            continue
        avg_diff = sum((1 - sim_matrix[i][k]) for k in range(num_rows) if k != i) / (num_rows - 1)
        imp = pop * avg_diff
        importance_values.append(imp)

    
    df = df.withColumn("importance", lit(0.0).cast(DoubleType()))
    for idx, imp in enumerate(importance_values):
        df = df.withColumn(
            "importance",
            when(
                (col(df.columns[0]) == data[idx][0]),  # simple ID match
                lit(imp)
            ).otherwise(col("importance"))
        )

    # --- Select top T tuples ---
    top_tuples = df.orderBy(col("importance").desc()).limit(T)
    top_tuples.drop("popularity", "importance").write.csv(output_file, header=True)

    print(f"Saved top {T} tuples to {output_file}")
    spark.stop()

