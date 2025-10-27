import os
import random
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max

"""
    Generate queries with weighted selection 
    (70% important, 25% second tier, 5% unimportant)
    Makes txt files of queries based on the input CSV file
    Returns the path to the generated queries file
"""
def generate_queries_weighted(spark, csv_file_path, num_queries, max_conditions):
 
    # Read the CSV file using Spark
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # Define column groups
    important_columns = ['Income', 'Year_Birth', 'Recency', 'NumDealsPurchases', 'NumStorePurchases', 'Education', 'Marital_Status', 'MntGoldProds', 'Complain']
    second_tier_columns = ['Kidhome', 'Teenhome', 'MntFruits', 'MntMeatProducts', 'NumWebPurchases', 'NumWebVisitsMonth', 'MntFishProducts', 'MntSweetProducts', 'NumCatalogPurchases', ]
    unimportant_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'ID', 'Dt_Customer', 'MntWines', 'Z_CostContact', 'Z_Revenue']

    
    # Get all numeric columns from the dataset for fallback
    all_numeric_columns = [col_name for col_name, dtype in df.dtypes if dtype in ['int', 'bigint', 'double', 'float']]
    
    # Combine all available columns
    all_columns = important_columns + second_tier_columns + unimportant_columns
    # Filter to only include columns that actually exist in the dataset
    available_columns = [col for col in all_columns if col in df.columns]
    
    # If we don't have enough columns, add from all numeric columns
    if len(available_columns) < 5:
        additional_cols = [col for col in all_numeric_columns if col not in available_columns]
        available_columns.extend(additional_cols[:5])  # Add up to 5 additional columns

    # Collect value ranges for all available columns using Spark
    numeric_ranges = {}
    for col_name in available_columns:
        if col_name in df.columns:
            min_max_df = df.select(
                min(col(col_name)).alias('min_val'),
                max(col(col_name)).alias('max_val')
            ).collect()[0]

            numeric_ranges[col_name] = (min_max_df['min_val'], min_max_df['max_val'])
            print(f"{col_name} range: {numeric_ranges[col_name]}")

    logical_operators = ['AND', 'OR']
    logical_weights = [0.5, 0.5]
    comparison_operators = ['=', '>', '<', '>=', '<=']
    comparison_weights = [0.2, 0.3, 0.3, 0.1, 0.1]

    queries = []
    for i in range(num_queries):
        num_conditions = random.randint(1, max_conditions)
        conditions = []
        selected_columns = []

        # Weighted column selection WITHOUT duplicates
        current_available = available_columns.copy()
        for _ in range(num_conditions):
            if len(current_available) == 0:
                break
                
            rand_val = random.random()
            if rand_val < 0.50:  # 50% chance for important columns
                available_important = [col for col in current_available if col in important_columns]
                if available_important:
                    chosen_col = random.choice(available_important)
                else:
                    chosen_col = random.choice(current_available)
            elif rand_val < 0.85:  # 35% chance for second tier columns (50 + 35 = 85)
                available_second = [col for col in current_available if col in second_tier_columns]
                if available_second:
                    chosen_col = random.choice(available_second)
                else:
                    chosen_col = random.choice(current_available)
            else:  # 5% chance for unimportant columns
                available_unimportant = [col for col in current_available if col in unimportant_columns]
                if available_unimportant:
                    chosen_col = random.choice(available_unimportant)
                else:
                    chosen_col = random.choice(current_available)
            
            selected_columns.append(chosen_col)
            # Remove the chosen column to prevent duplicates
            current_available.remove(chosen_col)

        # Choose operator when multiple conditions
        if num_conditions > 1:
            logical_op = random.choices(logical_operators, weights=logical_weights)[0]
        else:
            logical_op = None

        # Build conditions with comparison operators
        for j, col_name in enumerate(selected_columns):
            if col_name in numeric_ranges:
                min_val, max_val = numeric_ranges[col_name]

                # Choose comparison operator
                comp_op = random.choices(comparison_operators, weights=comparison_weights)[0]

                # Generate appropriate value based on operator
                if comp_op == '=':
                    value = random.randint(int(min_val), int(max_val))
                elif comp_op in ['>', '>=']:
                    # For greater than, pick value in lower range
                    threshold = int((min_val + max_val) * 0.3)
                    upper_bound = threshold if threshold < int(max_val) else int(max_val)
                    lower_bound = int(min_val)
                    if lower_bound <= upper_bound:
                        value = random.randint(lower_bound, upper_bound)
                    else:
                        value = random.randint(int(min_val), int(max_val))
                elif comp_op in ['<', '<=']:
                    # For less than, pick value in upper range
                    threshold = int((min_val + max_val) * 0.7)
                    lower_bound = threshold if threshold > int(min_val) else int(min_val)
                    upper_bound = int(max_val)
                    if lower_bound <= upper_bound:
                        value = random.randint(lower_bound, upper_bound)
                    else:
                        value = random.randint(int(min_val), int(max_val))
                else:
                    value = random.randint(int(min_val), int(max_val))

                # Build condition string
                condition = f"{col_name} {comp_op} {value}"
                conditions.append(condition)

        # Combine conditions
        if len(conditions) == 1:
            query = conditions[0]
        else:
            query = f" {logical_op} ".join(conditions)

        queries.append(query)

    # Save queries to file
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    queries_dir = os.path.dirname(csv_file_path)
    queries_file = os.path.join(queries_dir, f"{base_name}_queries.txt")
    with open(queries_file, "w") as file:
        file.write("\n".join(queries))

    print(f"Generated {len(queries)} queries with weighted column selection")
    print(f"Weight distribution: 70% important, 25% second tier, 5% unimportant")
    
    return queries_file
