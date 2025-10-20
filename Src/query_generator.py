import random
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max

def generate_queries_weighted(spark, csv_file_path, num_queries, max_conditions=3):
    """
    Generate queries with weighted selection of columns (70% important columns, 30% second tier columns)
    """
    # Read the CSV file using Spark
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    # Define column groups
    important_columns = ['Income', 'Year_Birth', 'Recency', 'NumDealsPurchases', 'NumStorePurchases']
    second_tier_columns = ['Kidhome', 'Teenhome', 'MntFruits', 'MntMeatProducts', 'NumWebPurchases', 'NumWebVisitsMonth']
    
    # Collect value ranges for all columns using Spark
    numeric_ranges = {}
    all_columns = important_columns + second_tier_columns
    
    for col_name in all_columns:
        if col_name in df.columns:
            min_max_df = df.select(
                min(col(col_name)).alias('min_val'),
                max(col(col_name)).alias('max_val')
            ).collect()[0]
            
            numeric_ranges[col_name] = (min_max_df['min_val'], min_max_df['max_val'])
            print(f"{col_name} range: {numeric_ranges[col_name]}")
    
    logical_operators = ['AND', 'OR']
    logical_weights = [0.7, 0.3] 
    comparison_operators = ['=', '>', '<', '>=', '<=']
    comparison_weights = [0.2, 0.3, 0.3, 0.1, 0.1]

    queries = []
    for i in range(num_queries):
        num_conditions = random.randint(1, max_conditions)
        conditions = []
        selected_columns = []
        
        # Weighted column selection
        for _ in range(num_conditions):
            if random.random() < 0.7:  # 70% chance for important columns
                selected_columns.append(random.choice(important_columns))
            else:  # 30% chance for second tier columns
                selected_columns.append(random.choice(second_tier_columns))
        
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
    with open("../Data/queries.txt", "w") as file:
        file.write("\n".join(queries))
    
    print(f"Generated {len(queries)} queries with weighted column selection")
    return queries

# Usage example:
if __name__ == "__main__":
    spark = SparkSession.builder.appName("QueryGen").getOrCreate()
    
    generate_queries_weighted(
        spark=spark,
        csv_file_path="../Data/tmp_small.csv",
        num_queries=10,
        max_conditions=3
    )
    
    spark.stop()