import pandas as pd

# MNLI
results = pd.read_csv('results/results.csv')
results = results[results['task'] == 'mnli']

# Keep just bias measure columns
working1 = results[['SEAT_gender', 'WEAT_gender', 'StereoSet_SS_gender',
                 'BiasNLI_NN', 'BiasNLI_FN', 'Matched Acc', 'Mismatched Acc']]

mnli_corr = working1.corr()

# STSB
results = pd.read_csv('results/results.csv')
results = results[results['task'] == 'stsb']

# Keep just bias measure columns
working1 = results[['SEAT_gender', 'WEAT_gender', 'StereoSet_SS_gender',
                 'BiasSTS', 'Spearmanr', 'Pearson']]

stsb_corr = working1.corr()