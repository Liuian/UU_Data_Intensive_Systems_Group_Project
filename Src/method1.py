from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
import sys

def parse_condition_part(condition_part):
    """
    Parse a single condition part into a Spark condition
    """
    condition_part = condition_part.strip()
    # Parse comparison operators
    if " >= " in condition_part:
        col_name, value = condition_part.split(" >= ")
        return col(col_name.strip()) >= int(value.strip())
    elif " <= " in condition_part:
        col_name, value = condition_part.split(" <= ")
        return col(col_name.strip()) <= int(value.strip())
    elif " > " in condition_part:
        col_name, value = condition_part.split(" > ")
        return col(col_name.strip()) > int(value.strip())
    elif " < " in condition_part:
        col_name, value = condition_part.split(" < ")
        return col(col_name.strip()) < int(value.strip())
    elif " = " in condition_part:
        col_name, value = condition_part.split(" = ")
        return col(col_name.strip()) == int(value.strip())
    return None

def parse_query_to_condition(query):
    """
    Parse a query string into a Spark condition
    """
    try:
        # Handle OR queries
        if " OR " in query:
            or_conditions = []
            for or_part in query.split(" OR "):
                and_condition = None
                # Handle AND conditions within each OR part
                for condition_part in or_part.split(" AND "):
                    condition = parse_condition_part(condition_part)
                    if condition is not None:
                        and_condition = condition if and_condition is None else and_condition & condition

                if and_condition is not None:
                    or_conditions.append(and_condition)

            # Combine OR conditions
            if or_conditions:
                condition = or_conditions[0]
                for or_cond in or_conditions[1:]:
                    condition = condition | or_cond
                return condition

        # Handle AND queries
        elif " AND " in query:
            condition = None
            for condition_part in query.split(" AND "):
                new_condition = parse_condition_part(condition_part)
                if new_condition is not None:
                    condition = new_condition if condition is None else condition & new_condition
            return condition

        # Handle single condition
        else:
            return parse_condition_part(query)

    except Exception as e:
        raise ValueError(f"Failed to parse query: {query}. Error: {e}")

    return None

def method1(spark, input_file, T, output_file, queries):
    # Read input data
    df = spark.read.csv(input_file, header=True, inferSchema=True)
    print(f"Loaded {df.count()} rows")

    # Start with popularity = 0
    df = df.withColumn("popularity", lit(0))

    # For each query, increment popularity for matching rows
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}: {query}")
        try:
            condition = parse_query_to_condition(query)

            if condition is not None:
                matching_count = df.filter(condition).count()
                print(f"  Rows matching: {matching_count}")
                df = df.withColumn("popularity",
                                  when(condition, col("popularity") + 1)
                                  .otherwise(col("popularity")))
            else:
                print(f"  No valid condition found")

        except Exception as e:
            print(f"Error processing query: {query}. Error: {e}")
            continue

    # Show popularity distribution
    print("\nPopularity distribution:")
    df.groupBy("popularity").count().orderBy("popularity").show()

    # Select top T tuples by popularity
    top_tuples = df.orderBy(col("popularity").desc()).limit(T)

    # Save results
    top_tuples.write.csv(output_file, header=True, mode='overwrite')

    print(f"Saved top {T} tuples to {output_file}")

