# create initial, empty results dataframe

import pandas as pd

results_df = pd.DataFrame(columns=['ID',
                                   'date',
                                   'seed',
                                   'task',  # MNLI or STS-B
                                   'pruning_method', # None if using base model
                                   'sparsity_level'
                                   'SEAT_before',
                                   'SEAT_after',
                                   'WEAT_before',
                                   'WEAT_after',
                                   'StereoSet_before',
                                   'StereoSet_after',
                                   'BiasNLI_NN_before',
                                   'BiasNLI_NN_after',
                                   'BiasNLI_FN_before',
                                   'BiasNLI_FN_after',
                                   'BiasSTS_before',
                                   'BiasSTS_after',
                                   'model_performance'
                                   # ??? pruning variables
                                   ])

results_df.to_csv('results.csv', index=False)
