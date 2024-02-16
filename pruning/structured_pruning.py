# Based on https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py

import argparse
import logging
import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm  # display progress bars during long-running operations
from utils import load_and_cache_examples, get_seed
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes, RobertaConfig, RobertaForSequenceClassification, \
    RobertaTokenizer
from utils import glue_compute_metrics as compute_metrics
from utils import get_device

logger = logging.getLogger(__name__)
logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True


# Used to log and visualize information about attention heads
def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


# Evaluate the importance of attention heads in a Transformer model by computing head importance scores.
def compute_heads_importance(args, model, eval_dataloader, compute_importance=True, head_mask=None):
    """ This method shows how to compute
        head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model.roberta.config.num_hidden_layers, model.roberta.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None and compute_importance:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)
        head_mask.requires_grad_(requires_grad=True)
    elif compute_importance:
        head_mask = head_mask.clone().detach()
        head_mask.requires_grad_(requires_grad=True)

    preds = None
    labels = None
    tot_tokens = 0.0
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(
            input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, head_mask=head_mask
        )
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        loss.backward()  # Backpropagate to populate the gradients in the head mask

        # compute head importance
        if compute_importance:
            head_importance += head_mask.grad.abs().detach()
            head_mask.grad.zero_()  # set gradients to zero to not overestimate importance of early batches

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)

        tot_tokens += input_mask.float().detach().sum().data

    if compute_importance:
        # Normalize
        head_importance /= tot_tokens
        # Layerwise importance normalization
        if not args.dont_normalize_importance_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        if not args.dont_normalize_global_importance:
            head_importance = (head_importance - head_importance.min()) / (
                    head_importance.max() - head_importance.min())

        # Print/save matrices
        np.save(os.path.join(args.output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())

        logger.info("Head importance scores")
        print_2d_tensor(head_importance)
        logger.info("Head ranked by importance scores")
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=args.device
        )
        head_ranks = head_ranks.view_as(head_importance)
        print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels


# masks specific attention heads in a Transformer model based on their importance scores.
def mask_heads(args, model, eval_dataloader):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    _, head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader)
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    original_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]
    logger.info("Pruning: original score: %f, threshold: %f", original_score, original_score * args.masking_threshold)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))

    current_score = original_score
    i = 0
    while current_score >= original_score * args.masking_threshold:
        head_mask = new_head_mask.clone()  # save current head mask
        if args.save_mask_all_iterations:
            np.save(os.path.join(args.output_dir, f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
            np.save(os.path.join(args.output_dir, f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())
        i += 1
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        selected_heads_to_mask = []
        for head in current_heads_to_mask:
            if len(selected_heads_to_mask) == num_to_mask or head_importance.view(-1)[head.item()] == float("Inf"):
                break
            layer_idx = head.item() // model.roberta.config.num_attention_heads
            head_idx = head.item() % model.roberta.config.num_attention_heads
            new_head_mask[layer_idx][head_idx] = 0.0
            selected_heads_to_mask.append(head.item())

        if not selected_heads_to_mask:
            break

        logger.info("Heads to mask: %s", str(selected_heads_to_mask))

        # new_head_mask = new_head_mask.view_as(head_mask)
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            args, model, eval_dataloader, head_mask=new_head_mask
        )
        preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        current_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]
        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )

    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask


# Pruning involves removing the weights of masked attention heads, effectively reducing the model's size and
# complexity. compares the performance of the pruned model to the masked model on the evaluation dataset. compares
# the execution time of the two models to assess the impact of pruning on computational efficiency.
def prune_heads(args, model, eval_dataloader, head_mask):
    """ This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args, model, eval_dataloader, compute_importance=False, head_mask=head_mask
    )
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    score_masking = compute_metrics(args.task_name, preds, labels)[args.metric_name]
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
        args, model, eval_dataloader, compute_importance=False, head_mask=None
    )
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    score_pruning = compute_metrics(args.task_name, preds, labels)[args.metric_name]
    new_time = datetime.now() - before_time

    logger.info(
        "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
        original_num_params,
        pruned_num_params,
        pruned_num_params / original_num_params * 100,
    )
    logger.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    logger.info("Pruning: speed ratio (new timing / original timing): %f percents", original_time / new_time * 100)


def structured_pruning(model, tokenizer, seed, task, device):
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
    metric_name = {
        "mnli": "acc",
        "sts-b": "corr",
    }[task]
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Prepare dataset
    val_data = load_and_cache_examples(args, task, tokenizer, evaluate=True)
    true_eval_len = len(val_data)
    # use subset of data if needed for debugging
    # subset_size = 100
    # eval_data = Subset(val_data, list(range(min(subset_size, len(val_data)))))
    eval_sampler = SequentialSampler(val_data) if args.local_rank == -1 else DistributedSampler(val_data)
    eval_dataloader = DataLoader(val_data, sampler=eval_sampler, batch_size=1)

    # perform pruning
    head_mask = mask_heads(args, model, eval_dataloader)
    prune_heads(args, model, eval_dataloader, head_mask)
    pruned_model_dir = f'{args.output_dir}/pruned_model/'
    if not os.path.exists(pruned_model_dir):
        os.makedirs(pruned_model_dir)
    model.save_pretrained(pruned_model_dir)
    tokenizer.save_pretrained(pruned_model_dir)