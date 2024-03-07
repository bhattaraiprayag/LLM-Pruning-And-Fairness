import pandas as pd

# Loading in the info about each run
def get_id_info(filepath):
    # Read in current results data frame
    results_df = pd.read_csv(filepath)
    # Keep just the columns of interest
    new_df = results_df[['ID', 'date', 'device', 'seed', 'task', 'pruning_method', 'sparsity_level', 'model_no', 'masking_threshold']]
    return new_df

# Get info about one specific run
def run_info(run_no):
    # Get all run info
    id_info = get_id_info('results/results.csv')

    # Filter to just the correct row
    id_row = id_info[id_info['ID']==run_no]

    return(id_row.to_dict('records')[0])

# Compose info about a run into a sentence
def run_phrase(dict_in):
    task = dict_in['task'].upper()
    model_no = dict_in['model_no']
    if pd.isna(dict_in['pruning_method']):
        action = 'without pruning'
    else:
        pruning_method = dict_in['pruning_method']
        sparsity = dict_in['sparsity_level']
        action = f'after {pruning_method} with {sparsity} sparsity'

    return f'This is the result for {task} model {model_no} {action}.'
