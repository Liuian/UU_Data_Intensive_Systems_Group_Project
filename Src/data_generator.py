# data_sampler.py
import pandas as pd
import random
import os
import glob

def create_subsets(small=100, medium=500, large=2000, input_file="../Data/marketing_campaign_converted.csv"):
    """
    Simple function to create 3 subsets from ../Data/marketing_campaign_converted.csv
    """
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    # Read the CSV file
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")

    # Create subsets
    subsets = [
        ("small", small),
        ("medium", medium), 
        ("large", large)
    ]
    
    created_files = []
    
    for size_name, sample_size in subsets:
        # Skip if sample size is larger than dataset
        if sample_size > len(df):
            print(f"Skipping {size_name}: sample size {sample_size} > dataset size {len(df)}")
            continue
        
        # Create output filename
        output_file = f"../Data/tmp_{size_name}.csv"
        
        # Create random sample
        sample_df = df.sample(n=sample_size, random_state=42)
        
        # Save to CSV
        sample_df.to_csv(output_file, index=False)
        created_files.append(output_file)
        print(f"Created {output_file} with {len(sample_df)} rows")

    print(f"\nCreated {len(created_files)} subset files:")
    for file in created_files:
        print(f"{file}")

    return created_files

if __name__ == "__main__":
    create_subsets()