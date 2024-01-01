# create initial, empty results dataframe

import pandas as pd

results_df = pd.DataFrame(columns=['ID',
                                   'date',
                                   'seed',
                                   'task',  # MNLI or STS-B
                                   'pruning_method',
                                   'SEAT_before',
                                   'SEAT_after',
                                   'WEAT_before',
                                   'WEAT_after',
                                   'StereoSet_before',
                                   'StereoSet_after',
                                   'BiasNLI_before',
                                   'BiasNLI_after',
                                   'BiasSTS_before',
                                   'BiasSTS_after',
                                   # ??? pruning variables
                                   ])

results_df.to_csv('results.csv', index=False)
