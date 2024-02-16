# Based on https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py

import argparse
import logging
import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from utils import load_examples, get_seed, compute_heads_importance, mask_heads
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from utils import glue_compute_metrics as compute_metrics
from utils import get_device

logger = logging.getLogger(__name__)
logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True



# Pruning involves removing the weights of masked attention heads, effectively reducing the model's size and
# complexity. compares the performance of the pruned model to the masked model on the evaluation dataset. compares
# the execution time of the two models to assess the impact of pruning on computational efficiency.
def prune_heads(model, eval_dataloader, device, local_rank, output_dir, task, head_mask):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    metric_name = {
        "mnli": "acc",
        "sts-b": "corr",
    }[task]
    output_mode = output_modes[task]

    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        model, eval_dataloader, device, local_rank, output_dir, compute_importance=False, head_mask=head_mask
    )
    preds = np.argmax(preds, axis=1) if output_mode == "classification" else np.squeeze(preds)
    score_masking = compute_metrics(task, preds, labels)[metric_name]
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask
    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    logger.info(f"{heads_to_prune}")
    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        model, eval_dataloader, device, local_rank, output_dir, compute_importance=False, head_mask=None
    )
    preds = np.argmax(preds, axis=1) if output_mode == "classification" else np.squeeze(preds)
    score_pruning = compute_metrics(task, preds, labels)[metric_name]
    new_time = datetime.now() - before_time

    logger.info(
        "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
        original_num_params,
        pruned_num_params,
        pruned_num_params / original_num_params * 100,
    )
    logger.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    logger.info("Pruning: speed ratio (new timing / original timing): %f percents", original_time / new_time * 100)


def structured_pruning(model, tokenizer, seed, task, device, masking_amount, masking_threshold):
    # Setup devices and distributed training
    local_rank = device
    device = get_device()
    n_gpu = 1
    torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(device, n_gpu, bool(local_rank != -1)))

    # Set seed
    get_seed(seed)

    # change task name stsb to sts-b (necessary for predefined functions)
    if task == "stsb":
        task = "sts-b"

    # set different variables related to task

    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Prepare dataset
    data_dir =
    val_data = load_examples(task, tokenizer, data_dir)
    # use subset of data if needed for debugging
    # subset_size = 100
    # eval_data = Subset(val_data, list(range(min(subset_size, len(val_data)))))
    eval_sampler = SequentialSampler(val_data) if args.local_rank == -1 else DistributedSampler(val_data)
    eval_dataloader = DataLoader(val_data, sampler=eval_sampler, batch_size=1)

    # set output directory
    output_dir =

    # perform pruning
    head_mask = mask_heads(model, eval_dataloader, device, local_rank, output_dir, task, masking_amount, masking_threshold)
    prune_heads(model, eval_dataloader, device, local_rank, output_dir, task, head_mask)