import pandas as pd

# Loading in the info about each run
def get_id_info(filepath):
    # Read in current results data frame
    results_df = pd.read_csv(filepath)
    # Keep just the columns of interest
    new_df = results_df[['ID', 'date', 'device', 'seed', 'task', 'pruning_method', 'sparsity_level', 'model_no']]
    return new_df

def run_info(run_no):
    # Get all run info
    id_info = get_id_info('results/results.csv')

    # Filter to just the correct row
    id_row = id_info[id_info['ID']==run_no]

    return(id_row.to_dict('records')[0])
