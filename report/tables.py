import pandas as pd

### Table for Bias NLI base model analysis

# Load data
data_bnli = pd.read_csv('results/run1/bias_nli.csv')
# Group by premise-hypothesis pairs
group_bnli = data_bnli.groupby(['premise_filler_word', 'hypothesis_filler_word'], as_index=False)[
    ['entailment', 'neutral', 'contradiction']].mean()

# Keep most extreme values for each gendered word - NOTE CONTRADICTION AND ENTAILMENT CURRENTLY SAVED THE WRONG WAY AROUND
entail_bnli = group_bnli.sort_values('contradiction', ascending=False).drop_duplicates(['hypothesis_filler_word'])
contradict_bnli = group_bnli.sort_values('entailment', ascending=False).drop_duplicates(['hypothesis_filler_word'])
neutral_bnli = group_bnli.sort_values('neutral', ascending=False).drop_duplicates(['hypothesis_filler_word'])

# Final table
output_bnli = entail_bnli[['hypothesis_filler_word', 'premise_filler_word']].copy()
output_bnli.rename(columns={'premise_filler_word': 'Entailment'}, inplace=True)
output_bnli = output_bnli.merge(neutral_bnli[['hypothesis_filler_word', 'premise_filler_word']], how='left', on='hypothesis_filler_word')
output_bnli.rename(columns={'premise_filler_word': 'Neutral'}, inplace=True)
output_bnli = output_bnli.merge(contradict_bnli[['hypothesis_filler_word', 'premise_filler_word']], how='left', on='hypothesis_filler_word')
output_bnli.rename(columns={'premise_filler_word': 'Contradiction', 'hypothesis_filler_word':'Gendered word'}, inplace=True)
