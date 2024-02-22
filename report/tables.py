import numpy as np
import pandas as pd

# Table for Bias NLI base model analysis
data_bnli = pd.read_csv('results/run1/bias_nli.csv')

group_bnli = data_bnli.groupby(['premise_filler_word', 'hypothesis_filler_word'], as_index=False)[['entailment', 'neutral', 'contradiction']].mean()
