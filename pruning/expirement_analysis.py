import numpy as np
import pathlib
import matplotlib.pyplot as plt
import time
import json
import itertools
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff

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
            mlp_importance = np.load(seed_dir / "mlp_importance.npy")
            mlp_data[task_dir.stem][seed_dir.stem] = {
                "mlp_mask": mlp_mask,
                "mlp_importance": mlp_importance
            }
    return mlp_data

experiments_path = pathlib.Path("../masks/heads_mlps")
heads = load_head_data(experiments_path)

experiments_path = pathlib.Path("../masks/heads_mlps_hans")
hans_heads = load_head_data(experiments_path)

for k, v in hans_heads.items():
    heads[k] = v

experiments_path = pathlib.Path("../masks/heads_mlps")
mlps = load_mlp_data(experiments_path)




experiments_path = pathlib.Path("../masks/heads_mlps_hans")
hans_mlps = load_mlp_data(experiments_path)

for k, v in hans_mlps.items():
    mlps[k] = v


def cochrans_q_masks(masks):
    inp = np.array(masks).transpose()
    return cochrans_q(inp)


def krippendorff_alpha_tasks_separate(data, mask="head_mask"):
    for task in sorted(data.keys()):
        krippendorff_alpha_tasks(data, [task], mask)


def krippendorff_alpha_tasks(data, tasks, mask="head_mask"):
    masks = []
    seeds = sorted(data[tasks[0]].keys())
    for task in tasks:
        for seed in seeds:
            masks.append(data[task][seed][mask].reshape(-1))
    alpha = krippendorff.alpha(masks)
    print(','.join(tasks))
    print("---------")
    print(f'alpha: {alpha}')


def fleiss_kappa_tasks_separate(data, mask="head_mask"):
    for task in sorted(data.keys()):
        fleiss_kappa_tasks(data, [task], mask)


def fleiss_kappa_tasks(data, tasks, mask="head_mask"):
    masks = []

    good_mask_sum = np.zeros_like(data["MNLI"]["seed_42"][mask].reshape(-1))
    bad_mask_sum = np.zeros_like(data["MNLI"]["seed_42"][mask].reshape(-1))
    seeds = sorted(data[tasks[0]].keys())
    for task in tasks:
        for seed in seeds:
            good_mask_sum = good_mask_sum + data[task][seed][mask].reshape(-1)
            bad_mask_sum = bad_mask_sum + (1 - data[task][seed][mask].reshape(-1))
    table = np.array([good_mask_sum, bad_mask_sum]).transpose()
    kappa = fleiss_kappa(table)
    print(','.join(tasks))
    print("---------")
    print(f'Feliss Kappa: {kappa}')


def print_p_value_tasks_separate(data, mask="head_mask"):
    for task in sorted(data.keys()):
        print_p_value_tasks(data, [task], mask)


def print_p_value_tasks(data, tasks, mask="head_mask"):
    masks = []
    seeds = sorted(data[tasks[0]].keys())
    for task in tasks:
        for seed in seeds:
            masks.append(data[task][seed][mask].reshape(-1))
    test_result = cochrans_q_masks(masks)
    print(','.join(tasks))
    print("---------------------------------------------------------------------")
    print(f'p-value: {test_result.pvalue}')
    print(
        f"{'Null hypothesis (all seeds are similar) is rejected.' if test_result.pvalue < 0.05 else 'Null hypothesis (all seeds are similar) is not rejected.'}")

    if len(tasks) == 1:
        masks_combos = list(itertools.combinations(range(len(masks)), 2))
    else:
        masks_combos = []
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                if i < j:
                    mask_1_idx = i
                    mask_2_idx = len(seeds) + j
                    masks_combos.append((mask_1_idx, mask_2_idx))
    similar_masks_combos = []
    for mask_1, mask_2 in masks_combos:
        r = cochrans_q_masks([masks[mask_1], masks[mask_2]])
        if r.pvalue >= 0.05:
            task1_name, seed1_name = tasks[mask_1 // len(seeds)], seeds[mask_1 % len(seeds)]
            task2_name, seed2_name = tasks[mask_2 // len(seeds)], seeds[mask_2 % len(seeds)]
            similar_masks_combos.append((f"{task1_name}-{seed1_name}", f"{task2_name}-{seed2_name}"))

    print(f"Total mask pairs where Null hypothesis is not rejected - {len(similar_masks_combos)}")
    print(f"Total mask pairs - {len(masks_combos)}")
    print(f"Percentage - {len(similar_masks_combos) / len(masks_combos)}")
    print("\nSimilar Mask Pairs:\n")
    print("\t".join([",".join(p) for p in similar_masks_combos]))
    print("\n\n")

### Seeds in a Task
## Heads

print_p_value_tasks_separate(heads)

krippendorff_alpha_tasks_separate(heads)

fleiss_kappa_tasks_separate(heads)

##MLPs

print_p_value_tasks_separate(mlps, mask="mlp_mask")

krippendorff_alpha_tasks_separate(mlps, mask="mlp_mask")

### Pairwise Task to task comparison
## Heads

tasks = sorted(heads.keys())
for t1, t2 in itertools.combinations(tasks, 2):
    print_p_value_tasks(heads, [t1, t2])

tasks = sorted(heads.keys())
for t1, t2 in itertools.combinations(tasks, 2):
    krippendorff_alpha_tasks(heads, [t1, t2])

##MLPs

for t1, t2 in itertools.combinations(tasks, 2):
    print_p_value_tasks(mlps, [t1, t2], mask="mlp_mask")


tasks = sorted(mlps.keys())
for t1, t2 in itertools.combinations(tasks, 2):
    krippendorff_alpha_tasks(mlps, [t1, t2], mask="mlp_mask")

sets = []
for seed in heads['MNLI']:
    a = heads['MNLI'][seed]['head_mask'].reshape(-1)
    sets.append({x[0] for x in np.argwhere(a == 1)})

union =  set.union(*sets)
intersection = set.intersection(*sets)
len(intersection) / len(union)