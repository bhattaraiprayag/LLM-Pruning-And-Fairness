# run this script to create initial, empty results dataframe

import pandas as pd

results_df = pd.DataFrame(columns=['ID',
                                   'date',
                                   'device',
                                   'seed',
                                   'task',  # MNLI or STS-B
                                   'pruning_method', # None if using base model
                                   'sparsity_level',
                                   'temperature',
                                   'SEAT_gender',
                                   'WEAT_gender',
                                   'StereoSet_LM_gender',
                                   'StereoSet_SS_gender',
                                   'BiasNLI_NN',
                                   'BiasNLI_FN',
                                   'BiasSTS',
                                   # ??? more pruning variables
                                   ])

results_df.to_csv('results.csv', index=False)
