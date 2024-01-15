import report.plots as plots
import pandas as pd

results = pd.read_csv('./results/results.csv')
results.sort_values('sparsity_level', inplace=True)
plots.all_plots(results)