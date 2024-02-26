import pandas as pd
from utils import run_info, run_phrase
import re


### Table for overview of all MNLI results
def mnli_overview(filepath):
    # Read in data
    results = pd.read_csv(filepath)
    results = results[results['task'] == 'mnli']

    # Group just for structured pruning, where the sparsity needs to be averaged
    working1 = results[results['pruning_method'] == 'structured']
    working1 = (working1.groupby(['masking_threshold', 'pruning_method'], as_index=False)[
                    ['sparsity_level', 'SEAT_gender', 'WEAT_gender', 'StereoSet_LM_gender', 'StereoSet_SS_gender',
                     'BiasNLI_NN', 'BiasNLI_FN', 'Matched Acc', 'Mismatched Acc']].mean())

    # Group for everything else, where the target sparsity was an input
    working2 = results[results['pruning_method'] != 'structured']
    working2 = (working2.groupby(['sparsity_level', 'pruning_method'], as_index=False)[
                    ['SEAT_gender', 'WEAT_gender', 'StereoSet_LM_gender', 'StereoSet_SS_gender',
                     'BiasNLI_NN', 'BiasNLI_FN', 'Matched Acc', 'Mismatched Acc']].mean())

    # Combine into a single table
    output = pd.concat([working1, working2], axis=0, ignore_index=True)
    # Reorder columns
    output = output[['pruning_method', 'sparsity_level', 'masking_threshold', 'Matched Acc', 'Mismatched Acc',
                     'BiasNLI_NN', 'BiasNLI_FN', 'SEAT_gender', 'WEAT_gender', 'StereoSet_LM_gender',
                     'StereoSet_SS_gender']]
    # Rename columns
    output.rename(columns={'pruning_method': 'Pruning method', 'sparsity_level': 'Sparsity level',
                           'masking_threshold': 'Masking threshold', 'Matched Acc': 'Matched accuracy',
                           'Mismatched Acc': 'Mismatched accuracy', 'BiasNLI_NN': 'Bias-NLI NN',
                           'BiasNLI_FN': 'Bias-NLI FN', 'SEAT_gender': 'SEAT', 'WEAT_gender': 'WEAT',
                           'StereoSet_LM_gender': 'StereoSet LM', 'StereoSet_SS_gender': 'StereoSet SS'}, inplace=True)
    # Sort rows
    output.sort_values(by=['Pruning method', 'Sparsity level'], inplace=True)

    # Convert to latex
    latex = output.to_latex(index=False,
                            column_format='p{0.16\\textwidth}p{0.06\\textwidth}p{0.07\\textwidth}p{0.07\\textwidth}p{0.07\\textwidth}p{0.07\\textwidth}p{0.07\\textwidth}p{0.04\\textwidth}p{0.04\\textwidth}p{0.07\\textwidth}p{0.07\\textwidth}',
                            label=f'tab:mnli_all',
                            caption=f'Results from the MNLI models. Where the masking threshold was specified for structured pruning, the average sparsity level is shown.',
                            na_rep='-',
                            float_format="%.3f")
    # Change to table* so it is page wide instead of confined to column
    latex = latex.replace('table', 'table*')
    # Save the LaTeX output
    with open(f"report/tables/mnli.tex", "w") as f:
        f.write(latex)


### Table for overview of all STS-B results
def stsb_overview(filepath):
    results = pd.read_csv(filepath)
    results = results[results['task'] == 'stsb']

    # Group just for structured pruning, where the sparsity needs to be averaged
    working1 = results[results['pruning_method'] == 'structured']
    working1 = (working1.groupby(['masking_threshold', 'pruning_method'], as_index=False)[
                    ['sparsity_level', 'SEAT_gender', 'WEAT_gender', 'StereoSet_LM_gender', 'StereoSet_SS_gender',
                     'BiasSTS', 'Spearmanr', 'Pearson']].mean())

    # Group for everything else, where the target sparsity was an input
    working2 = results[results['pruning_method'] != 'structured']
    working2 = (working2.groupby(['sparsity_level', 'pruning_method'], as_index=False)[
                    ['SEAT_gender', 'WEAT_gender', 'StereoSet_LM_gender', 'StereoSet_SS_gender',
                     'BiasSTS', 'Spearmanr', 'Pearson']].mean())

    # Combine into a single table
    output = pd.concat([working1, working2], axis=0, ignore_index=True)
    # Reorder columns
    output = output[['pruning_method', 'sparsity_level', 'masking_threshold', 'Spearmanr', 'Pearson',
                     'BiasSTS', 'SEAT_gender', 'WEAT_gender', 'StereoSet_LM_gender', 'StereoSet_SS_gender']]
    # Rename columns
    output.rename(columns={'pruning_method': 'Pruning method', 'sparsity_level': 'Sparsity level',
                           'masking_threshold': 'Masking threshold', 'Spearmanr': 'Spearman rank',
                           'BiasSTS': 'Bias-STS', 'SEAT_gender': 'SEAT', 'WEAT_gender': 'WEAT',
                           'StereoSet_LM_gender': 'StereoSet LM', 'StereoSet_SS_gender': 'StereoSet SS'}, inplace=True)
    # Sort rows
    output.sort_values(by=['Pruning method', 'Sparsity level'], inplace=True)

    # Convert to latex
    latex = output.to_latex(index=False,
                            column_format='p{0.16\\textwidth}p{0.06\\textwidth}p{0.07\\textwidth}p{0.07\\textwidth}p{0.07\\textwidth}p{0.07\\textwidth}p{0.04\\textwidth}p{0.04\\textwidth}p{0.07\\textwidth}p{0.07\\textwidth}',
                            label=f'tab:stsb_all',
                            caption=f'Results from the STSB models. Where the masking threshold was specified for structured pruning, the average sparsity level is shown.',
                            na_rep='-',
                            float_format="%.3f")
    # Change to table* so it is page wide instead of confined to column
    latex = latex.replace('table', 'table*')
    # Save the LaTeX output
    with open(f"report/tables/stsb.tex", "w") as f:
        f.write(latex)


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
    output_bnli = output_bnli.merge(neutral_bnli[['hypothesis_filler_word', 'premise_filler_word']], how='left',
                                    on='hypothesis_filler_word')
    output_bnli.rename(columns={'premise_filler_word': 'Neutral'}, inplace=True)
    output_bnli = output_bnli.merge(contradict_bnli[['hypothesis_filler_word', 'premise_filler_word']], how='left',
                                    on='hypothesis_filler_word')
    output_bnli.rename(columns={'premise_filler_word': 'Contradiction', 'hypothesis_filler_word': 'Gendered word'},
                       inplace=True)

    latex = output_bnli.to_latex(index=False,
                                 column_format='lccc',
                                 label=f'tab:bnli{run_no}',
                                 caption=f'Results from Bias-NLI for the job titles that most entail or contradict gendered words. {info_sent}')
    # Save the LaTeX output
    with open(f"report/tables/bnli{run_no}.tex", "w") as f:
        f.write(latex)


### Results and table for Bias STS base model analysis
def bsts_results(run_no):
    # Get info about run
    info_run = run_info(run_no)
    info_sent = run_phrase(info_run)

    # Load data
    data_bsts = pd.read_csv(f'results/run{run_no}/bias_sts.csv')

    # Descriptive statistics
    print(data_bsts['abs_diff'].describe())
    print(data_bsts['score_male'].describe())
    print(data_bsts['score_female'].describe())

    # Get sentence pairs with the highest abs_diff
    row = data_bsts.iloc[[data_bsts['abs_diff'].idxmax()]]
    print(f'Male Sentence: {row.iloc[0]["sentence_occupation"]}')
    print(f'Male Sentence: {row.iloc[0]["sentence_male"]} - Score: {row.iloc[0]["score_male"]}')
    print(f'Female Sentence: {row.iloc[0]["sentence_female"]} - Score: {row.iloc[0]["score_female"]}')
    print(f'Absolute difference: {row.iloc[0]["abs_diff"]}')

    # Get top 10 of male and female occupations
    group_bsts = data_bsts.groupby('occupation', as_index=False)['diff'].mean()
    top_male_occupations = group_bsts.sort_values(by='diff', ascending=False).head(10)
    top_male_occupations.rename(columns={'diff': 'Difference between similarity scores',
                                         'occupation': 'Occupation'}, inplace=True)
    top_female_occupations = group_bsts.sort_values(by='diff', ascending=True).head(10)
    top_female_occupations.rename(columns={'diff': 'Difference between similarity scores',
                                           'occupation': 'Occupation'}, inplace=True)

    latex_male = top_male_occupations.to_latex(index=False,
                                               float_format="%.2f",
                                               column_format='lr',
                                               label=f'tab:bsts{run_no}_male',
                                               caption=f'Top 10 stereotypically male occupations based on the results of Bias-STS. The difference between similarity scores is calculated with (male - female). {info_sent}')

    latex_female = top_female_occupations.to_latex(index=False,
                                                   float_format="%.2f",
                                                   column_format='lr',
                                                   label=f'tab:bsts{run_no}_male',
                                                   caption=f'Top 10 stereotypically female occupations based on the results of Bias-STS. The difference between similarity scores is calculated with (male - female). {info_sent}')
    # Save the LaTeX outputs
    with open(f"report/tables/bsts{run_no}_male.tex", "w") as f:
        f.write(latex_male)
    with open(f"report/tables/bsts{run_no}_female.tex", "w") as f:
        f.write(latex_female)
