import argparse
import logging
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, SequentialSampler, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from scratch import MODEL_CLASSES
from scratch import ALL_MODELS
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from experiment_impact_tracker.compute_tracker import ImpactTracker
#from some_module import compute_heads_importance, compute_metrics

#from run_glue import ALL_MODELS, MODEL_CLASSES, load_and_cache_examples, set_seed
from glue_metrics import glue_compute_metrics as compute_metrics

#from model_bert import BertForSequenceClassification
#from config_bert import BertConfig

# Instantiate the model and configuration
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
config = RobertaConfig.from_pretrained('roberta-base')

logger = logging.getLogger(__name__)
logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True

pruning_method = prune.ln_structured
#Step 1: It provides valuable insights into the distribution and importance of attention across different heads. prune heads with the lowest entrop

def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)
#Step2 : Used to log and visualize information about attention heads

def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))

#Step 3: Iteration computing the loss, logits, and attention values.
# (Backpropagation is performed to populate the gradients in the head mask (used for computing head importance scores))
# STS-B dataset includes label_scores instead of label_ids, as STS-B involves predicting similarity scores
# rather than discrete labels. Adjustments might still be necessary depending on your specific dataset and requirements.
# Make sure to check the tokenization and data loading process to ensure compatibility with your STS-B dataset.
def compute_heads_importance_stsb(args, model, eval_dataloader, compute_entropy=True, compute_importance=True,
                                  head_mask=None):
    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
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
        input_ids, attention_mask, label_scores = batch  # STS-B has label_scores instead of label_ids

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(input_ids, attention_mask=attention_mask, labels=label_scores, head_mask=head_mask)
        loss, logits, all_attentions = outputs.loss, outputs.logits, outputs.attentions

        loss.backward()  # Backpropagate to populate the gradients in the head mask

        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * attention_mask.float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_scores.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_scores.detach().cpu().numpy(), axis=0)

        tot_tokens += attention_mask.float().detach().sum().data

    if compute_entropy:
        # Normalize
        attn_entropy /= tot_tokens
        np.save(os.path.join(args.output_dir, "attn_entropy.npy"), attn_entropy.detach().cpu().numpy())
        print("Attention Entropies:")
        print_2d_tensor(attn_entropy)

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
        print("Head Importance Scores:")
        print_2d_tensor(head_importance)

        # Print head ranks by importance scores
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=args.device
        )
        head_ranks = head_ranks.view_as(head_importance)
        print("Head Ranks by Importance Scores:")
        print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels

#Step 4: Implements a process of pruning (masking) attention heads in a transformer model based on their importance scores. This pruning is done iteratively, removing heads with the lowest importance scores until a certain threshold is reached.

def compute_heads_importance(args, model, eval_dataloader, compute_entropy, head_mask):
    pass


def mask_heads_roberta(args, model, eval_dataloader):
    # Compute initial importance scores
    head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False)
    preds = np.squeeze(preds)
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

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            args, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
        )
        preds = np.squeeze(preds)
        current_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]
        logger.info(
            "Masking: current score: %f, remaining heads %d (%.1f percent)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )

    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances."
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Whether to overwrite data in output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--save_mask_all_iterations", action="store_true", help="Saves the masks and importance scores in all iterations"
    )
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )

    parser.add_argument(
        "--try_masking", action="store_true", help="Whether to try to mask head until a threshold of accuracy."
    )
    parser.add_argument(
        "--use_train_data", action="store_true", help="Use training set for computing masks"
    )
    parser.add_argument(
        "--masking_threshold",
        default=0.9,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument("--metric_name", default=None, type=str, help="Metric to use for head masking.")

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, sequences shorter padded.",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    args = parser.parse_args()
