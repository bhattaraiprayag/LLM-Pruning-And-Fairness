import pandas as pd


# Loading in the info about each run
def get_id_info(filepath):
    # Read in current results data frame
    results_df = pd.read_csv(filepath)
    # Keep just the columns of interest
    new_df = results_df[['ID', 'date', 'device', 'seed', 'task', 'pruning_method', 'sparsity_level', 'temperature']]
    return new_df