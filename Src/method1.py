from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, sum, when
import sys

def method1(input_file, T, output_file, queries):
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Method1") \
        .getOrCreate()
    
    # Read input data
    df = spark.read.csv(input_file, header=True, inferSchema=True)
   
    # Start with popularity = 0
    df = df.withColumn("popularity", lit(0))
    
    # For each query, increment popularity for matching rows
    for query in queries:
        try: 
            condition = None
            for condition_part in query.split(" AND "):
                col_name, value = condition_part.split(" = ")
                col_name = col_name.strip()
                value = int(value.strip())
                
                if condition is None:
                    condition = (col(col_name) == value)
                else:
                    condition = condition & (col(col_name) == value)
            
            if condition is not None:
                df = df.withColumn("popularity", 
                                  when(condition, col("popularity") + 1)
                                  .otherwise(col("popularity")))
        except:
            continue  # Skip invalid queries
     # Select top T tuples by popularity
    top_tuples = df.orderBy(col("popularity").desc()).limit(T)
    
    # Save results (without popularity column)
    top_tuples.drop("popularity").write.csv(output_file, header=True)
    
    print(f"Saved top {T} tuples to {output_file}")
    spark.stop()
