import matplotlib.pyplot as plt
from randomization_analysis import plot_all_task_metrics , analyzed_metrics
import numpy as np

from randomization_analysis import plot_all_task_metrics, analyzed_metrics


#Load data related to heads from the directory and then the data is organized into a nested dictionary structure with tasks, seeds, and corresponding masks.
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

# It takes (experiments_path) and loads (MLP) data from that directory.
# It returns a nested dictionary (mlp_data) containing information about the MLP masks for each task and seed.
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

def plot_all_task_metrics(metrics, title):
    for name in metrics[0][1].keys():
        plot_all_task_metrics(analyzed_metrics, name, title)


def plot_matrix(mean_matrix, stddv_matrix, x_labels, y_labels, save_path, x_title='', y_title='', colour_map=plt.cm.OrRd):
    fig, ax = plt.subplots(figsize=(9,8))
    ax.set_xlabel(x_title, labelpad=20)
    ax.set_ylabel(y_title)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.imshow(mean_matrix, cmap=colour_map)
    plt.tight_layout()
    for i in range(len(mean_matrix)):
        for j in range(len(mean_matrix[0])):
            mean = mean_matrix[i, j]
            std =  stddv_matrix[i, j]
            ax.text(j, i, f"{mean:.2f}\n{std:.2f}", va='center', ha='center')
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

    def show_heads_heatmap(head_data, save_name):
        tasks = sorted(list(head_data.keys()))
        seeds = head_data[tasks[0]].keys()

        #     pruned_head_results = np.zeros((len(tasks) * len(seeds), 12, 12)) # 12 layers with 12 heads each
        #     i = 0
        #     for seed_idx, seed in enumerate(seeds):
        #         for task_idx, task in enumerate(tasks):
        #             pruned_head_results[i] = head_data[task][seed]["head_mask"]
        #             i += 1

        pruned_head_results = np.zeros((len(seeds), 12, 12))  # 12 layers with 12 heads each
        for task_idx, task in enumerate(tasks):
            for seed_idx, seed in enumerate(seeds):
                pruned_head_results[seed_idx] += head_data[task][seed]["head_mask"]
        mean_pruned_heads = np.mean(pruned_head_results, axis=0)
        std_pruned_heads = np.std(pruned_head_results, axis=0, ddof=1)
        plot_matrix(mean_pruned_heads, std_pruned_heads, list(map(str, range(12))), list(map(str, range(12))),
                    save_name, "Head", "Layer")
        print("mean_survival_mean", mean_pruned_heads.mean())
        print("mean_survival_std", std_pruned_heads.mean())
        print("stable heads", (mean_pruned_heads > 5).sum())


def show_mlps_heatmap(mlp_data, save_name):
    tasks = sorted(list(mlp_data.keys()))
    seeds = mlp_data[tasks[0]].keys()

    #     pruned_mlps_results = np.zeros((len(tasks) * len(seeds), 1, 12)) # 12 layers with 12 heads each
    #     i = 0
    #     for seed_idx, seed in enumerate(seeds):
    #         for task_idx, task in enumerate(tasks):
    #             pruned_mlps_results[i] = mlp_data[task][seed]["mlp_mask"]
    #             i += 1

    pruned_mlps_results = np.zeros((len(seeds), 1, 12))  # 12 layers
    for task_idx, task in enumerate(tasks):
        for seed_idx, seed in enumerate(seeds):
            pruned_mlps_results[seed_idx] += mlp_data[task][seed]["mlp_mask"]
    mean_pruned_mlps = np.mean(pruned_mlps_results, axis=0)
    std_pruned_mlps = np.std(pruned_mlps_results, axis=0, ddof=1)
    plot_matrix(mean_pruned_mlps, std_pruned_mlps, list(map(str, range(12))), [], save_name, "Layer", "")
    print("mean_survival_mean", mean_pruned_mlps.mean())
    print("mean_survival_std", std_pruned_mlps.mean())
    print("stable mlps", (mean_pruned_mlps > 5).sum())

def show_task_specific_heads_heatmap(head_data, directory, save_suffix):
    tasks = sorted(list(head_data.keys()))
    seeds = head_data[tasks[0]].keys()
    for task in tasks:
        print(task)
        pruned_head_results = np.zeros((len(seeds), 12, 12)) # 12 layers with 12 heads each
        for seed_idx, seed in enumerate(seeds):
            pruned_head_results[seed_idx] += head_data[task][seed]["head_mask"]
        mean_pruned_heads = np.mean(pruned_head_results, axis=0)
        std_pruned_heads = np.std(pruned_head_results, axis=0, ddof=1)

        plot_matrix(mean_pruned_heads, std_pruned_heads, list(map(str, range(12))), list(map(str, range(12))), f"{directory}/{task}_{save_suffix}", "Head", "Layer")
        print("mean_survival_mean", mean_pruned_heads.mean())
        print("mean_survival_std", std_pruned_heads.mean())
        print("stable heads", (mean_pruned_heads == 1.0).sum())


def show_task_specific_mlp_heatmap(mlp_data, directory, save_suffix):
    tasks = sorted(list(mlp_data.keys()))
    seeds = mlp_data[tasks[0]].keys()
    for task in tasks:
        print(task)
        pruned_mlps_results = np.zeros((len(seeds), 1, 12))  # 12 layers
        for seed_idx, seed in enumerate(seeds):
            pruned_mlps_results[seed_idx] += mlp_data[task][seed]["mlp_mask"]
        mean_pruned_mlps = np.mean(pruned_mlps_results, axis=0)
        std_pruned_mlps = np.std(pruned_mlps_results, axis=0, ddof=1)

        plot_matrix(mean_pruned_mlps, std_pruned_mlps, list(map(str, range(12))), [],
                    f"{directory}/{task}_{save_suffix}", "Layer", "")

        print("mean_survival_mean", mean_pruned_mlps.mean())
        print("mean_survival_std", std_pruned_mlps.mean())
        print("stable mlps", (mean_pruned_mlps == 1.0).sum())

#experiments_path = pathlib.Path("../masks/heads")
#separate_head_data = load_head_data(experiments_path)

#xperiments_path = pathlib.Path("../masks/heads")
#separate_head_data = load_head_data(experiments_path)

#experiments_path = pathlib.Path("../masks/mlps")
#separate_mlp_data = load_mlp_data(experiments_path)

#experiments_path = pathlib.Path("../masks/heads_mlps")
#together_head_data = load_head_data(experiments_path)

#experiments_path = pathlib.Path("../masks/heads_mlps")
#together_mlp_data = load_mlp_data(experiments_path)

#experiments_path = pathlib.Path("../masks/heads_mlps_super/")
#super_head_data = load_head_data(experiments_path)
#super_mlp_data = load_mlp_data(experiments_path)

##Heads Heat map - Separate
#show_heads_heatmap(separate_head_data, "heatmaps/all_tasks/head_heatmap_separate.pdf")

##Heads Heat map - Separate
#show_heads_heatmap(separate_head_data, "heatmaps/all_tasks/head_heatmap_separate.pdf")

##Heads Heat map - Separate
#show_heads_heatmap(separate_head_data, "heatmaps/all_tasks/head_heatmap_separate.pdf")

##MLPs Heat map - Together setting
#show_mlps_heatmap(together_mlp_data, "heatmaps/all_tasks/mlp_heatmap_together.pdf")


##Task Specific Heads Heat Map - Together setting
#show_task_specific_heads_heatmap(together_head_data, "heatmaps/task_specific", "head_heatmap_together.pdf")