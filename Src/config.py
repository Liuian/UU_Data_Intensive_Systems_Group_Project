FILE_PATH = "./Data/marketing_campaign_cleaned.csv"

# for testing only, small size
RESULTS_PATH = "./Results/experiments_results.csv"
REP_RANGE = 2 # 5
QUERIES_NUM_LIST = [50]
CONDITIONS_LIST = [1]   
DATASET_SIZE_T_VALUES = [(5, 1)] 
METHODS_TO_RUN = ["2"]  # ["1", "2", "3"]
SUBSET_SIZES = [5, 10, 20]

# Experiment 1: For method 1, 2 and 3
EXP1_RESULTS_PATH = "./Results/experiments_results_exp1.csv"
EXP1_REP_RANGE = 5
EXP1_QUERIES_NUM_LIST = [50, 200]
EXP1_CONDITIONS_LIST = [1, 3]
EXP1_DATASET_SIZE_T_VALUES = [(4, 1), (8, 2), (12, 3)]    # (dataset_size, T) pairs   # 
EXP1_METHODS_TO_RUN = ["1", "2", "3"]   
EXP1_SUBSET_SIZES = [4, 8, 12]    

# Experiment 2: For method 1 and 2
EXP2_RESULTS_PATH = "./Results/experiments_results_exp2.csv"
EXP2_REP_RANGE = 5
EXP2_QUERIES_NUM_LIST = [100, 200]
EXP2_CONDITIONS_LIST = [1, 3]
EXP2_DATASET_SIZE_T_VALUES = [(100, 20), (500, 100), (2000, 400)]    # (dataset_size, T) pairs
EXP2_METHODS_TO_RUN = ["1", "2"]
EXP2_SUBSET_SIZES = [100, 500, 2000]