import pathlib
import json
import numpy as np
import time
import matplotlib.pyplot as plt

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from compute_super_masks import load_head_data
from compute_super_masks import load_head_data
from component_heatmaps import load_head_data

def plot_matrix(mean_matrix, stddv_matrix, x_labels, y_labels, save_path, x_title='', y_title='',
                colour_map=plt.cm.OrRd):
    mask = np.tri(mean_matrix.shape[0], k=-1).T
    mean_matrix = np.ma.array(mean_matrix, mask=mask)
    stddv_matrix = np.ma.array(stddv_matrix, mask=mask)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlabel(x_title, labelpad=20)
    ax.set_ylabel(y_title)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    mask = np.tri(mean_matrix.shape[0], k=0).T  # Leave Diagonal Uncoloured
    mean_matrix_wo_diagonal = np.ma.array(mean_matrix, mask=mask)
    ax.imshow(mean_matrix_wo_diagonal, cmap=colour_map)
    plt.gca().xaxis.tick_bottom()

    mean_mean = np.ma.mean(mean_matrix_wo_diagonal)
    for i in range(len(mean_matrix)):
        for j in range(len(mean_matrix[0])):
            if i >= j:
                mean = mean_matrix[i, j]
                text = f"{mean:.2f}"
                if stddv_matrix.sum() != 0:
                    std = stddv_matrix[i, j]
                    text += f"\n{std:.2f}"
                color = 'black' if mean < mean_mean or i == j else 'white'
                ax.text(j, i, text, va='center', ha='center', color=color)
    if save_path:
        fig.savefig(save_path)

def load_head_data(experiments_path):
    head_data = {}
    for task_dir in experiments_path.iterdir():
        head_data[task_dir.stem] = {}
        for seed_dir in task_dir.iterdir():
            head_mask = np.load(seed_dir / "head_mask.npy")
            head_data[task_dir.stem][seed_dir.stem] = {
                "head_mask": head_mask,
            }
            return head_data

def load_mlp_data(experiments_path):
    mlp_data = {}
    for task_dir in experiments_path.iterdir():
        mlp_data[task_dir.stem] = {}
        for seed_dir in task_dir.iterdir():
            mlp_mask = np.load(seed_dir / "mlp_mask.npy")
            mlp_data[task_dir.stem][seed_dir.stem] = {
                "mlp_mask": mlp_mask,
            }
    return mlp_data


def task_overlap_attention_heads(head_data, save_name):
    tasks = sorted(list(head_data.keys()))
    seeds = head_data["MNLI"].keys()
    results = np.zeros((len(seeds), len(tasks), len(tasks)))
    for seed_idx, seed in enumerate(seeds):
        for task_1_idx, task_1 in enumerate(tasks):
            for task_2_idx, task_2 in enumerate(tasks):
                common_heads = np.sum(head_data[task_1][seed]["head_mask"] * head_data[task_2][seed]["head_mask"])
                results[seed_idx][task_1_idx][task_2_idx] = common_heads

    mean_shared_heads = np.mean(results, axis=0)
    std_shared_heads = np.std(results, axis=0, ddof=1)
    plot_matrix(mean_shared_heads, std_shared_heads, tasks, tasks, save_name, "Task", "Task", plt.cm.Purples)

def task_overlap_mlps(mlp_data, save_name):
    tasks = sorted(list(head_data.keys()))
    seeds = mlp_data["MNLI"].keys()
    results = np.zeros((len(seeds), len(tasks), len(tasks)))
    for seed_idx, seed in enumerate(seeds):
        for task_1_idx, task_1 in enumerate(tasks):
            for task_2_idx, task_2 in enumerate(tasks):
                common_mlps = np.sum(mlp_data[task_1][seed]["mlp_mask"] * mlp_data[task_2][seed]["mlp_mask"])
                results[seed_idx][task_1_idx][task_2_idx] = common_mlps

    mean_shared_mlps = np.mean(results, axis=0)
    std_shared_mlps = np.std(results, axis=0, ddof=1)
    std_shared_mlps = np.std(results, axis=0, ddof=1)
    plot_matrix(mean_shared_mlps, std_shared_mlps, tasks, tasks, save_name, "Task", "Task", plt.cm.Purples)

    #experiments_path = pathlib.Path("../masks/heads/")
    #head_data = load_head_data(experiments_path)

    #experiments_path = pathlib.Path("../masks/mlps/")
    #mlp_data = load_mlp_data(experiments_path)

    #experiments_path = pathlib.Path("../masks/heads_mlps/")
    #together_head_data = load_head_data(experiments_path)
    #together_mlp_data = load_mlp_data(experiments_path)

    #experiments_path = pathlib.Path("../masks/heads_mlps_super/")
    #super_head_data = load_head_data(experiments_path)
    #super_mlp_data = load_mlp_data(experiments_path)