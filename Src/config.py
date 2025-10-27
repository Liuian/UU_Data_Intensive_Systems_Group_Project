FILE_PATH = "./Data/marketing_campaign_cleaned.csv"
RESULTS_PATH = "./Results/experiments_results.csv"

# Define experiment parameter grid
REP_RANGE = 2 # 5
QUERIES_NUM_LIST = [50] # [50, 200]
CONDITIONS_LIST = [1]   # [1, 3]
DATASET_SIZE_T_VALUES = [(5, 1)]    # (dataset_size, T) pairs   # [(5, 1), (10, 2), (20, 4)]
METHODS_TO_RUN = ["3"]  # ["1", "2", "3"]
