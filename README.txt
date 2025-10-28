Project README

Overview
--------
This project runs a set of experiments that select top-T tuples from a marketing dataset using multiple methods. Each experiment produces per-run outputs and consolidated CSV summaries of all experiments.

How to run
----------
Run the experiment runner from the repository root:

        python ./Src/main.py

Where output files go
---------------------
- Per-run top-T results:
    - All per-run outputs are written under the `./Results` directory.
    - For each method run the code creates an output folder with the naming pattern:

        `./Results/m{method_name}_dsize{base_name}_q{qnum}_c{cond}_rep{rep}_seed{seed}`

        Inside that folder the actual CSV file with the selected top-T tuples (written by Spark) can be found (typically as a `part-*.csv` file).

- Consolidated experiment CSVs:
    - Two experiment summary CSVs are produced and saved in `./Results`:
        - `experiments_results_exp1.csv`
        - `experiments_results_exp2.csv`
    - These CSVs contain one row per run with metrics such as runtime, imp_R, average cosine similarity, diversity, and query coverage.

Configuration
-------------
All experiment parameters (dataset path, query counts, repetition ranges, which methods to run, subset sizes, and result paths) are defined in `./Src/config.py`.

Data source
-----------
This project expects the cleaned marketing dataset to be placed at the path referenced in `config.py` (the default is `./Data/marketing_campaign_cleaned.csv`). The original source dataset can be obtained from the Kaggle "customer personality analysis" dataset (marketing_campaign.csv).

Notes
-----
- The query generator writes query text files next to each source dataset; `main.py` passes a per-run seed so queries can be deterministic when the same seed is used.
- Spark writes CSV outputs as directories containing `part-*.csv` files; the runner searches those folders for the CSV file when reading results back.
