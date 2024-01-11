import pathlib
import json
import numpy as np
import time
import itertools
import sys
import torch
import tqdm

sys.path.insert(0, '../src')
from transformers import RobertaForSequenceClassification
from model_roberta import RobertaForSequenceClassification
from run_glue import add_masks
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sns.set()


def bessel_correct(stddev):
    return stddev * np.sqrt(5. / 4.)


def confidence_interval_t(stddev):
    t_95 = 2.776
    return bessel_correct(stddev) * t_95


def flatten_metrics(metrics):
    flattened = {}
    task_metric_metric_name = [
        ("CoLA", "mcc", "Matthews correlation"),
        ("MNLI", "mnli_acc", "Accuracy"),
        ("MRPC", "acc", "Accuracy"),
        ("QNLI", "acc", "Accuracy"),
        ("QQP", "acc", "Accuracy"),
        ("RTE", "acc", "Accuracy"),
        ("SST-2", "acc", "Accuracy"),
        ("STS-B", "pearson", "Pearson correlation"),
        ("WNLI", "acc", "Accuracy")
    ]
    for task, metric, metric_name in task_metric_metric_name:
        flattened[task] = {
            "metric": metric_name,
            "mean": metrics[task][metric][0],
            "stdv": bessel_correct(metrics[task][metric][1]),
            "ci": confidence_interval_t(metrics[task][metric][1])
        }

    return flattened
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def plot_all_task_metrics(metrics, save_path, task_weights_pct, bbox_to_anchor_left=0.6):
    all_tasks =  ["MNLI", "QNLI", "RTE", "MRPC", "QQP", "SST-2", "CoLA", "STS-B", "WNLI"]
    experiment_metrics = [(name, flatten_metrics(metrics)) for name, metrics in metrics]
    experiment_names = [e_m[0] for e_m in experiment_metrics]

    metrics_data = [e_m[1] for e_m in experiment_metrics]
    x_pos = np.arange(len(experiment_names))
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))

    # Get a color map
    my_cmap = sns.color_palette("Paired") + sns.color_palette("Paired")[:8]
    patterns = [r"*", r"|", r"\\", r"\\||", r"--", r"--||", r"//", r"//||", "xx", "xx||", "..", "..||", "oo", "oo||"]

    for i, task in enumerate(all_tasks):
        means = [m[task]["mean"] for m in metrics_data]
        errors = [m[task]["stdv"] for m in metrics_data]
        row = i // 3
        col = i % 3
        axs[row, col].bar(x_pos, means, yerr=errors, align='center',
                          color=my_cmap)  # my_cmap(my_norm(range(len(x_pos)))))
        axs[row, col].set_ylabel(metrics_data[0][task]["metric"])

        mean = task_weights_pct[task]['mean'] * 100
        std = task_weights_pct[task]['std'] * 100
        axs[row, col].set_title(f"{task} ({mean:.0f}%, std {std:.0f}%)")
        axs[row, col].set_xticks([])

        bars = axs[row, col].patches

        for bar, hatch in zip(bars, patterns):  # loop over bars and hatches to set hatches in correct order
            bar.set_hatch(hatch)
    legend_elements = [Patch(facecolor=my_cmap[i], hatch=patterns[i], label=exp) for i, exp in
                       enumerate(experiment_names)]
    l_col = 3
    legend = plt.legend(flip(legend_elements, l_col), flip(experiment_names, l_col), loc='best', ncol=l_col,
                        bbox_to_anchor=(bbox_to_anchor_left, -0.1), labelspacing=1.5, handlelength=4)
    for patch in legend.get_patches():
        patch.set_height(10)
        patch.set_y(-1)
    plt.subplots_adjust(right=1.5)
    fig.tight_layout()
    plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

def modules_pruned(model):
    parameters_to_prune = []
    for layer in model.bert.encoder.layer:
        parameters = [
            layer.attention.self.key,
            layer.attention.self.key,
            layer.attention.self.query,
            layer.attention.self.query,
            layer.attention.self.value,
            layer.attention.self.value,
            layer.attention.output.dense,
            layer.attention.output.dense,
            layer.intermediate.dense,
            layer.intermediate.dense,
            layer.output.dense,
        ]
        parameters_to_prune.extend(parameters)
    return parameters_to_prune


importance_subnet_sizes = {}
for task in ["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI", "QNLI", "RTE", "WNLI"]:

    task_weight_pct = []

    for seed in ["seed_1337", "seed_42", "seed_86", "seed_71", "seed_166"]:
        # Load Model
        model_path = f"../models/finetuned/{task}/{seed}/"
        transformer_model = RobertaForSequenceClassification.from_pretrained(model_path)
        total_before_prune = sum(p.numel() for p in transformer_model.parameters())

        # Prune
        mask_path = f"../masks/heads_mlps/{task}/{seed}/"
        head_mask = np.load(f"{mask_path}/head_mask.npy")
        mlp_mask = np.load(f"{mask_path}/mlp_mask.npy")
        head_mask = torch.from_numpy(head_mask)
        heads_to_prune = {}
        for layer in range(len(head_mask)):
            heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
            heads_to_prune[layer] = heads_to_mask
        mlps_to_prune = [h[0] for h in (1 - torch.from_numpy(mlp_mask).long()).nonzero().tolist()]

        transformer_model.prune_heads(heads_to_prune)
        transformer_model.prune_mlps(mlps_to_prune)
        transformer_model = transformer_model.eval()

        total_after_prune = sum(p.numel() for p in transformer_model.parameters())
        task_weight_pct.append((total_after_prune / total_before_prune) - 0.21)

    importance_subnet_sizes[task] = {
        'mean': np.mean(task_weight_pct),
        'std': np.std(task_weight_pct, ddof=1)
    }

importance_subnet_super_sizes = {}
for task in ["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI", "QNLI", "RTE", "WNLI"]:

    task_weight_pct = []

    for seed in ["seed_1337", "seed_42", "seed_86", "seed_71", "seed_166"]:
        # Load Model
        model_path = f"../models/finetuned/{task}/{seed}/"
        transformer_model = RobertaForSequenceClassification.from_pretrained(model_path)
        total_before_prune = sum(p.numel() for p in transformer_model.parameters())

        # Prune
        mask_path = f"../masks/heads_mlps_super/{task}/{seed}/"
        head_mask = np.load(f"{mask_path}/head_mask.npy")
        mlp_mask = np.load(f"{mask_path}/mlp_mask.npy")
        head_mask = torch.from_numpy(head_mask)
        heads_to_prune = {}
        for layer in range(len(head_mask)):
            heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
            heads_to_prune[layer] = heads_to_mask
        mlps_to_prune = [h[0] for h in (1 - torch.from_numpy(mlp_mask).long()).nonzero().tolist()]

        transformer_model.prune_heads(heads_to_prune)
        transformer_model.prune_mlps(mlps_to_prune)
        transformer_model = transformer_model.eval()

        total_after_prune = sum(p.numel() for p in transformer_model.parameters())
        task_weight_pct.append((total_after_prune / total_before_prune) - 0.21)

    importance_subnet_super_sizes[task] = {
        'mean': np.mean(task_weight_pct),
        'std': np.std(task_weight_pct, ddof=1)
    }


def modules_pruned(model):
    parameters_to_prune = []
    for layer in model.bert.encoder.layer:
        parameters = [
            layer.attention.self.key,
            layer.attention.self.key,
            layer.attention.self.query,
            layer.attention.self.query,
            layer.attention.self.value,
            layer.attention.self.value,
            layer.attention.output.dense,
            layer.attention.output.dense,
            layer.intermediate.dense,
            layer.intermediate.dense,
            layer.output.dense,
        ]
        parameters_to_prune.extend(parameters)
    return parameters_to_prune


magnitude_subnet_sizes = {}

for task in ["CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI", "QNLI", "RTE", "WNLI"]:

    task_weight_pct = []

    for seed in ["seed_1337", "seed_42", "seed_86", "seed_71", "seed_166"]:
        # Load Model
        model_path = f"../models/finetuned/{task}/{seed}/"
        transformer_model = RobertaForSequenceClassification.from_pretrained(model_path)
        total_before_prune = sum(p.numel() for p in transformer_model.parameters())

        # Prune
        mask_path = f"../masks/global/{task}/{seed}/magnitude_mask.p"
        masks = torch.load(mask_path)
        add_masks(transformer_model, masks)
        transformer_model = transformer_model.eval()

        total_after_prune = 0
        for n, p in transformer_model.named_parameters():
            if "_orig" in n:
                params = masks[n[:-5] + "_mask"].sum()
            else:
                params = p.numel()
            total_after_prune += params

        task_weight_pct.append((total_after_prune / total_before_prune) - 0.21)  # Remove embedding %

    magnitude_subnet_sizes[task] = {
        'mean': np.mean(task_weight_pct),
        'std': np.std(task_weight_pct, ddof=1),
        evaluation_dir : pathlib.Path("../evaluate_masked")}

    name_path = {

        "majority baseline": evaluation_dir / "freq_baseline" / "results.json",
        "full model": evaluation_dir / "no_mask" / "baseline" / "results.json",
        "'good' subnetwork (pruned)": evaluation_dir / "head_mlp" / "baseline" / "results.json",
        "'good' subnetwork (retrained)": evaluation_dir / "head_mlp_retrained" / "baseline" / "results.json",
        "random subnetwork (pruned)": evaluation_dir / "head_mlp_random" / "baseline" / "results.json",
        "random subnetwork (retrained)": evaluation_dir / "head_mlp_retrained" / "baseline" / "results.json",
        "'bad' subnetwork (pruned)": evaluation_dir / "head_mlp_bad_1" / "baseline" / "results.json",
        "'bad' subnetwork (retrained)": evaluation_dir / "head_mlp_bad_1_retrained" / "baseline" / "results.json" }

    analyzed_metrics = []
    for name, path in name_path.items():
        with path.open() as f:
         metrics = json.load(f)
    analyzed_metrics.append((name, metrics))
    plot_all_task_metrics(analyzed_metrics, "evaluation/importance_pruning_evaluation.pdf", importance_subnet_sizes)

    evaluation_dir = pathlib.Path("../evaluate_masked")
    name_path = {

        "majority baseline": evaluation_dir / "freq_baseline" / "results.json",
        # "1 head mlp baseline  (retrained)": evaluation_dir / "head_mlp_zero_retrained" / "baseline" / "results.json",
        "full model": evaluation_dir / "no_mask" / "baseline" / "results.json",

        "'good' subnetwork (pruned)": evaluation_dir / "head_mlp_super" / "baseline" / "results.json",
        "'good' subnetwork (retrained)": evaluation_dir / "head_mlp_super_retrained" / "baseline" / "results.json",
        "random subnetwork (pruned)": evaluation_dir / "head_mlp_super_midling" / "baseline" / "results.json",
        "random subnetwork (retrained)": evaluation_dir / "head_mlp_super_midling_retrained" / "baseline" / "results.json",
        "'bad' subnetwork (pruned)": evaluation_dir / "head_mlp_super_bizzaro" / "baseline" / "results.json",
        "'bad' subnetwork (retrained)": evaluation_dir / "head_mlp_super_bizzaro_retrained" / "baseline" / "results.json",
    }
analyzed_metricsnalyzed_metrics = []
for name, path in name_path.items():
    with path.open() as f:
     metrics = json.load(f)
    analyzed_metrics.append((name, metrics))
    plot_all_task_metrics(analyzed_metrics, "evaluation/importance_pruning_super_evaluation.pdf",
                          importance_subnet_super_sizes)
