import numpy as np
import pandas as pd

### Table for Bias NLI base model analysis

# Load data
data_bnli = pd.read_csv('results/run1/bias_nli.csv')
# Group by premise-hypothesis pairs
group_bnli = data_bnli.groupby(['premise_filler_word', 'hypothesis_filler_word'], as_index=False)[
    ['entailment', 'neutral', 'contradiction']].mean()

# Keep most extreme values for each gendered word
entail_bnli = group_bnli.sort_values('entailment', ascending=False).drop_duplicates(['hypothesis_filler_word'])
contradict_bnli = group_bnli.sort_values('contradiction', ascending=False).drop_duplicates(['hypothesis_filler_word'])
neutral_bnli = group_bnli.sort_values('neutral', ascending=False).drop_duplicates(['hypothesis_filler_word'])