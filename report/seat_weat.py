import os
import json
import pandas as pd

def get_id_info(filepath):
    # Read in current results data frame
    results_df = pd.read_csv(filepath)
    # Keep just the columns of interest
    new_df = results_df[['ID', 'date', 'device', 'seed', 'task', 'pruning_method', 'sparsity_level', 'temperature']]
    return new_df

def main():
    new_df = get_id_info('results/results.csv')

    for i in new_df['ID']:
        with open(f'results/run{i}/seatandweat_aggregated.json', 'r') as f:
            json_file = json.load(f)
        print(i)


if __name__ == "__main__":
    main()