import os
import random
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, min, max

def output_file_name(input_file_path, output_prefix, suffix):
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_file = os.path.join(output_prefix, f"{base_name}_{suffix}")
    return output_file

def compute_diversity_from_avg_pairwise_cosine(vecs):
    """Compute average pairwise cosine similarity for rows in vecs.
    Returns 1.0 for single-item input, None for empty input.
    """
    if vecs is None or len(vecs) == 0:
        return None
    m = len(vecs)
    if m == 1:
        return 1.0
    pair_sims = []
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
    diversity = 1.0 - float(np.mean(pair_sims))

    return diversity

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

def create_subsets(small=100, medium=500, large=2000, input_file=""):
    """
    Create three subset CSVs from the provided input CSV.

    Output files are written into the same directory as `input_file` (so they
    live next to the source dataset). This avoids hardcoded ../Data paths and
    makes the function robust to different working directories.
    """
    # Validate input path
    if not input_file:
        print("Error: input_file not provided to create_subsets()")
        return
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # Create subsets
    subsets = [
        ("small", small),
        ("medium", medium), 
        ("large", large)
    ]
    
    created_files = []
    
    # Place subset files in the same folder as the input file
    input_dir = os.path.dirname(os.path.abspath(input_file)) or "."

    for size_name, sample_size in subsets:
        # Skip if sample size is larger than dataset
        if sample_size > len(df):
            print(f"Skipping {size_name}: sample size {sample_size} > dataset size {len(df)}")
            continue
        # Create output filename next to the input file
        output_dir = input_dir
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{sample_size}.csv")

        # Create random sample
        sample_df = df.sample(n=sample_size, random_state=42)

        # Save to CSV (ensure parent directory exists)
        sample_df.to_csv(output_file, index=False)
        created_files.append(output_file)
        print(f"Created {output_file} with {len(sample_df)} rows")

    print(f"\nCreated {len(created_files)} subset files:")
    for file in created_files:
        print(f"{file}")

    return created_files

def generate_queries_weighted(spark, csv_file_path, num_queries, max_conditions, seed=None):
    """
        Generate queries with weighted selection 
        (70% important, 25% second tier, 5% unimportant)
        Makes txt files of queries based on the input CSV file
        Returns the path to the generated queries file
    """
 
    # Read the CSV file using Spark
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # Define column groups
    important_columns = ['Income', 'Year_Birth', 'Recency', 'NumDealsPurchases', 'NumStorePurchases', 'Education', 'Marital_Status', 'MntGoldProds', 'Complain']
    second_tier_columns = ['Kidhome', 'Teenhome', 'MntFruits', 'MntMeatProducts', 'NumWebPurchases', 'NumWebVisitsMonth', 'MntFishProducts', 'MntSweetProducts', 'NumCatalogPurchases', ]
    unimportant_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'ID', 'Dt_Customer', 'MntWines', 'Z_CostContact', 'Z_Revenue']

    
    # Optional deterministic RNG: if seed is provided, use a local Random instance
    rng = random if seed is None else random.Random(seed)

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
        num_conditions = rng.randint(1, max_conditions)
        conditions = []
        selected_columns = []

        # Weighted column selection WITHOUT duplicates
        current_available = available_columns.copy()
        for _ in range(num_conditions):
            if len(current_available) == 0:
                break
                
            rand_val = rng.random()
            if rand_val < 0.50:  # 50% chance for important columns
                available_important = [col for col in current_available if col in important_columns]
                if available_important:
                    chosen_col = rng.choice(available_important)
                else:
                    chosen_col = rng.choice(current_available)
            elif rand_val < 0.85:  # 35% chance for second tier columns (50 + 35 = 85)
                available_second = [col for col in current_available if col in second_tier_columns]
                if available_second:
                    chosen_col = rng.choice(available_second)
                else:
                    chosen_col = rng.choice(current_available)
            else:  # 5% chance for unimportant columns
                available_unimportant = [col for col in current_available if col in unimportant_columns]
                if available_unimportant:
                    chosen_col = rng.choice(available_unimportant)
                else:
                    chosen_col = rng.choice(current_available)
            
            selected_columns.append(chosen_col)
            # Remove the chosen column to prevent duplicates
            current_available.remove(chosen_col)

        # Choose operator when multiple conditions
        if num_conditions > 1:
            logical_op = rng.choices(logical_operators, weights=logical_weights)[0]
        else:
            logical_op = None

        # Build conditions with comparison operators
        for j, col_name in enumerate(selected_columns):
            if col_name in numeric_ranges:
                min_val, max_val = numeric_ranges[col_name]

                # Choose comparison operator
                comp_op = rng.choices(comparison_operators, weights=comparison_weights)[0]

                # Generate appropriate value based on operator
                if comp_op == '=':
                    value = rng.randint(int(min_val), int(max_val))
                elif comp_op in ['>', '>=']:
                    # For greater than, pick value in lower range
                    threshold = int((min_val + max_val) * 0.3)
                    upper_bound = threshold if threshold < int(max_val) else int(max_val)
                    lower_bound = int(min_val)
                    if lower_bound <= upper_bound:
                        value = rng.randint(lower_bound, upper_bound)
                    else:
                        value = rng.randint(int(min_val), int(max_val))
                elif comp_op in ['<', '<=']:
                    # For less than, pick value in upper range
                    threshold = int((min_val + max_val) * 0.7)
                    lower_bound = threshold if threshold > int(min_val) else int(min_val)
                    upper_bound = int(max_val)
                    if lower_bound <= upper_bound:
                        value = rng.randint(lower_bound, upper_bound)
                    else:
                        value = rng.randint(int(min_val), int(max_val))
                else:
                    value = rng.randint(int(min_val), int(max_val))

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
    queries_file = output_file_name(csv_file_path, csv_file_path, "queries.txt")
    with open(queries_file, "w") as file:
        file.write("\n".join(queries))

    print(f"Generated {len(queries)} queries with weighted column selection")
    print(f"Weight distribution: 70% important, 25% second tier, 5% unimportant")
    
    return queries_file



# Load queries from text file
def load_queries(queries_file):
    with open(queries_file, 'r') as file:
        queries = [line.strip() for line in file if line.strip()]
    print(f"Loaded {len(queries)} queries from {queries_file}")
    return queries
