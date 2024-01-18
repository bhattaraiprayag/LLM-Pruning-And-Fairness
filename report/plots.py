import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in data
#results = pd.read_csv('./results/results.csv')
#results.sort_values('sparsity_level', inplace=True)

# Filter pandas dataframe to task and pruning method specific, also including unpruned model
def results_filter(results, task, pruning_method):
    output = results.query('task==@task and (pruning_method==@pruning_method or sparsity_level==0)')
    return output

# Create a plot where all the pruning methods are included for some metric, changing over sparsity levels
def all_prune_figure(results, task, measure, values=[0,1,0.99]):
    # The first 2 entries of values set the y-axis, the other is for the optimum to add a green line
    pruning_methods = ['l1-unstructured', 'l1-unstructured-linear']
    for pm in pruning_methods:
        data = results_filter(results, task, pm)
        plt.plot(data['sparsity_level'], data[measure])
    plt.title(f'{task} - {measure}')
    plt.xlabel("Sparsity level")
    plt.legend(pruning_methods)
    plt.ylim(values[0], values[1])
    plt.axhline(y=values[2], color='lightgreen', linestyle='dashed')
    plt.savefig(f'./report/figures/all_prune_{task}_{measure}.png')
    plt.close()

def all_performance_figure(results, task, measure, values=[0,1,0.99]):
    groups = results.groupby('pruning_method')
    for name, group in groups:
        plt.plot(group['Matched Acc'], group[measure], marker='o', linestyle='', markersize=12, label=name)
    plt.title(f'{task} - {measure}')
    plt.xlabel("Matched accuracy")
    plt.legend()
    plt.ylim(values[0], values[1])
    plt.axhline(y=values[2], color='lightgreen', linestyle='dashed')
    plt.savefig(f'./report/figures/all_performance_{task}_{measure}.png')
    plt.close()

def all_plots(results):
    all_prune_figure(results, 'mnli', 'seat_gender', [0,1,0.01])
    all_prune_figure(results, 'mnli', 'weat_gender', [0,1,0.01])
    all_prune_figure(results, 'mnli', 'Stereoset_LM_gender')
    all_prune_figure(results, 'mnli', 'Stereoset_SS_gender', [0,1,0.5])
    all_prune_figure(results, 'mnli', 'Matched Acc')
    all_prune_figure(results, 'mnli', 'Mismatched Acc')
    all_performance_figure(results, 'mnli', 'seat_gender', [0,1,0.01])
