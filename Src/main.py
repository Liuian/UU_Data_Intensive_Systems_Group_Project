import sys
import os
import pandas as pd
from data_generator import create_subsets
from query_generator import generate_queries

file_path = "../Data/marketing_campaign_converted.csv"
if os.path.exists(file_path):
    print(f"File {file_path} exists, proceeding to create subsets.")
create_subsets()
print("Created the subset files based on marketing_campaign_converted.csv")

generate_queries(10, 2)
