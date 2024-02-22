import pandas as pd
from utils import run_info, run_phrase

### Table for overview of all MNLI results
def mnli_overview(filepath):
    results = pd.read_csv(filepath)
    results = results[results['task']=='mnli']

    working = results[results['pruning_method']=='structured'].copy()
    working = (working.groupby(['masking_threshold', 'pruning_method'], as_index=False)[
        ['sparsity_level', 'SEAT_gender', 'WEAT_gender', 'StereoSet_LM_gender', 'StereoSet_SS_gender',
         'BiasNLI_NN', 'BiasNLI_NN', 'Matched Acc', 'Mismatched Acc']].mean())

    output = results[results['pruning_method']!='structured'].copy()
    output = (output.groupby(['sparsity_level', 'pruning_method'], as_index=False)[
                  ['SEAT_gender', 'WEAT_gender', 'StereoSet_LM_gender', 'StereoSet_SS_gender',
                   'BiasNLI_NN', 'BiasNLI_NN', 'Matched Acc', 'Mismatched Acc']].mean())


### Table for overview of all STS-B results
def stsb_overview(filepath):
    results = pd.read_csv(filepath)
    results = results[results['task']=='stsb']


### Table for Bias NLI base model analysis
def bnli_table(run_no):
    # Get info about run
    info_run = run_info(run_no)
    info_sent = run_phrase(info_run)

    # Load data
    data_bnli = pd.read_csv(f'results/run{run_no}/bias_nli.csv')
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

    latex = output_bnli.to_latex(index=False,
                                 column_format='lccc',
                                 label=f'tab:bnli{run_no}',
                                 caption=f'Results from Bias-NLI for the job titles that most entail or contradict gendered words. {info_sent}')
    # Save the LaTeX output
    with open(f"report/tables/bnli{run_no}.tex", "w") as f:
        f.write(latex)