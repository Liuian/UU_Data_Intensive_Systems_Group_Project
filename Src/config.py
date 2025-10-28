FILE_PATH = "./Data/marketing_campaign_cleaned.csv"

# Define experiment parameter grid
RESULTS_PATH = "./Results/experiments_results.csv"
REP_RANGE = 2 # 5
QUERIES_NUM_LIST = [50] # [50, 200]
CONDITIONS_LIST = [1]   # [1, 3]
DATASET_SIZE_T_VALUES = [(5, 1)]    # (dataset_size, T) pairs   # [(5, 1), (10, 2), (20, 4)]
METHODS_TO_RUN = ["3"]  # ["1", "2", "3"]
SUBSET_SIZES = [5, 10, 20]  # small, medium, large

# Experiment 1: For method 1, 2 and 3
EXP1_RESULTS_PATH = "./Results/experiments_results_exp1.csv"
EXP1_REP_RANGE = 5
EXP1_QUERIES_NUM_LIST = [50, 200]
EXP1_CONDITIONS_LIST = [1, 3]
EXP1_DATASET_SIZE_T_VALUES = [(5, 1), (10, 2), (20, 4)]    # (dataset_size, T) pairs   # 
EXP1_METHODS_TO_RUN = ["1", "2", "3"]
EXP1_SUBSET_SIZES = [5, 10, 20]

# Experiment 2: For method 1 and 2
EXP2_RESULTS_PATH = "./Results/experiments_results_exp2.csv"
EXP2_REP_RANGE = 5
EXP2_QUERIES_NUM_LIST = [200, 1000, 4000]
EXP2_CONDITIONS_LIST = [1, 3]
EXP2_DATASET_SIZE_T_VALUES = [(100, 20), (500, 100), (2000, 400)]    # (dataset_size, T) pairs
EXP2_METHODS_TO_RUN = ["2", "3"]
EXP2_SUBSET_SIZES = [100, 500, 2000]  # small, medium, large