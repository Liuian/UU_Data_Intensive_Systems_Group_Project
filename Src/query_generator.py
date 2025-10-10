import random

def generate_queries(num_queries, max_conditions=3):
    queries = []
    for _ in range(num_queries):
        num_conditions = random.randint(1, max_conditions)
        conditions = []
        selected_columns = random.sample(table_columns, num_conditions)
        
        for col in selected_columns:
            value = random.randint(1, 100)  # Match data range
            conditions.append(f"{col} = {value}")
        
        query = " AND ".join(conditions)
        queries.append(query)
    
 
    with open("../Data/queries.txt", "w") as file:
        file.write("\n".join(queries))
    print(queries)


table_columns = [
    'Year_Birth',           # Numeric years (1940-2000)
    'Income',               # Numeric income values
    'Kidhome',              # Discrete counts (0, 1, 2)
    'Teenhome',             # Discrete counts (0, 1, 2)
    'Recency',              # Numeric (0-100 range)
    'MntWines',             # Monetary amounts
    'MntFruits',            # Monetary amounts
    'MntMeatProducts',      # Monetary amounts
    'MntFishProducts',      # Monetary amounts
    'MntSweetProducts',     # Monetary amounts
    'MntGoldProds',         # Monetary amounts
    'NumDealsPurchases',    # Count data
    'NumWebPurchases',      # Count data
    'NumCatalogPurchases',  # Count data
    'NumStorePurchases',    # Count data
    'NumWebVisitsMonth',    # Count data
    'AcceptedCmp3',         # Binary (0 or 1)
    'AcceptedCmp4',         # Binary (0 or 1)
    'AcceptedCmp5',         # Binary (0 or 1)
    'AcceptedCmp1',         # Binary (0 or 1)
    'AcceptedCmp2',         # Binary (0 or 1)
    'Complain',             # Binary (0 or 1)
    'Response'              # Binary (0 or 1)
]