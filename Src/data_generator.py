# data_sampler.py
import pandas as pd
import random
import os
import glob

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

if __name__ == "__main__":
    create_subsets()