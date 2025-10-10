import sys
import os
import pandas as pd
from data_generator import create_subsets
from query_generator import generate_queries
from method1 import method1

def load_queries(file_path):
     # Reading generated queries
    with open(file_path, "r") as file:
        queries = file.readlines()
    # Remove newline characters
    queries = [line.strip() for line in queries]
    
    print(f"Loaded {len(queries)} queries")
    return queries

file_path = "../Data/marketing_campaign_converted.csv"
if os.path.exists(file_path):
    print(f"File {file_path} exists, proceeding to create subsets.")
create_subsets()
print("Created the subset files based on marketing_campaign_converted.csv")

generate_queries(10, 2)

method1("../Data/tmp_small.csv", 30, "../Results/tmp_m1_small.csv", load_queries("../Data/queries.txt"))

