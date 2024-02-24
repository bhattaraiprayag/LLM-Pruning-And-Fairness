import os
import logging
import random
import csv
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm  # display progress bars during long-running operations
from sklearn.metrics import matthews_corrcoef
from transformers import glue_processors as processors
from torch.utils.data import TensorDataset
from transformers.data.metrics import simple_accuracy, acc_and_f1, pearson_and_spearman
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import InputExample
from datasets import load_dataset, DatasetDict


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin"
}

def load_data_hub(task, model_no):
    if task == 'mnli':
        if model_no == 1:
            raw_datasets = load_dataset(
                "glue",
                'mnli',
                split=['train', 'validation_matched[:50%]',
                       'validation_mismatched[:50%]',
                       'validation_matched[-50%:]',
                       'validation_mismatched[-50%:]'
                       ]
            )
            datasets = DatasetDict({'train': raw_datasets[0],
                                        'validation_matched': raw_datasets[1],
                                        'validation_mismatched': raw_datasets[2],
                                        'test_matched': raw_datasets[3],
                                        'test_mismatched': raw_datasets[4]
                                        })
        else:
            raw_datasets = load_dataset(
                "glue",
                "mnli",
                split=['train+validation_matched', 'validation_mismatched[:50%]', 'validation_mismatched[-50%:]']
            )
            # 2.5% test_matched + validation_matched (keep the same ratio as in the original split)
            train_testvalid = raw_datasets[0].train_test_split(test_size=0.025, shuffle=True, seed=model_no)
            # Split test_matched + validation_matched in half test_matched, half validation_matched
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True, seed=model_no)
            # gather everything into a single DatasetDict
            datasets = DatasetDict({
                'train': train_testvalid['train'],
                'test_matched': test_valid['test'],
                'validation_matched': test_valid['train'],
                'test_mismatched': raw_datasets[1],
                'validation_mismatched': raw_datasets[2]
            })
    elif task == 'stsb':
        if model_no == 1:
            raw_datasets = load_dataset(
                "glue",
                'stsb',
                split=['train',
                       'validation[:50%]',
                       'validation[-50%:]'
                       ]
            )
            datasets = DatasetDict({'train': raw_datasets[0],
                                        'validation': raw_datasets[1],
                                        'test': raw_datasets[2]
                                        })
        else:
            raw_datasets = load_dataset(
                "glue",
                "stsb",
                split='train+validation'
            )
            # 20% test + validation (keep the same ratio as in the original split)
            train_testvalid = raw_datasets.train_test_split(test_size=0.2, shuffle=True, seed=model_no)
            # Split test + valid in half test, half valid
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True, seed=model_no)
            # gather everything into a single DatasetDict
            datasets = DatasetDict({
                'train': train_testvalid['train'],
                'test': test_valid['test'],
                'validation': test_valid['train']})

    return datasets

def get_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_sparsity(model):
    total_params = 0
    nonzero_params = 0
    layer_sparsity = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:  # exclude non-trainable parameters
            continue
        layer_size = param.numel()
        layer_nonzero = torch.count_nonzero(param)
        layer_sparsity[name] = 1 - layer_nonzero.item() / layer_size
        total_params += layer_size
        nonzero_params += layer_nonzero.item()
    overall_sparsity = 1 - nonzero_params / total_params
    return overall_sparsity


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


RobertaLayerNorm = torch.nn.LayerNorm

# Logger Setup
logger = logging.getLogger(__name__)


def print_2d_tensor(tensor):
    """ Print a 2D tensor (used to log and visualize information about attention heads)"""
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(model, eval_dataloader, device, local_rank, output_dir, compute_importance=True,
                             head_mask=None):
    """ This method shows how to compute
        head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model.roberta.config.num_hidden_layers, model.roberta.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(device)

    if head_mask is None and compute_importance:
        head_mask = torch.ones(n_layers, n_heads).to(device)
        head_mask.requires_grad_(requires_grad=True)
    elif compute_importance:
        head_mask = head_mask.clone().detach()
        head_mask.requires_grad_(requires_grad=True)

    preds = None
    labels = None
    tot_tokens = 0.0
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])):
        batch = tuple(t.to(device) for t in batch)
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
        # Layer-wise importance normalization
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        # Print/save matrices
        np.savetxt(os.path.join(output_dir, "head_importance.npy"), head_importance.detach().cpu().numpy())

        logger.info("Head importance scores")
        print_2d_tensor(head_importance)
        logger.info("Head ranked by importance scores")
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=device
        )
        head_ranks = head_ranks.view_as(head_importance)
        print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels


def mask_heads(model, eval_dataloader, device, local_rank, output_dir, task, masking_amount, masking_threshold):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    metric_name = {
        "mnli": "acc",
        "sts-b": "corr",
    }[task]
    output_mode = output_modes[task]

    _, head_importance, preds, labels = compute_heads_importance(model, eval_dataloader, device, local_rank, output_dir)
    preds = np.argmax(preds, axis=1) if output_mode == "classification" else np.squeeze(preds)
    original_score = glue_compute_metrics(task, preds, labels)[metric_name]
    logger.info("Pruning: original score: %f, threshold: %f", original_score, original_score * masking_threshold)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * masking_amount))

    current_score = original_score
    i = 0
    while current_score >= original_score * masking_threshold:
        head_mask = new_head_mask.clone()
        # save current head mask
        np.savetxt(os.path.join(output_dir, f"head_mask_{i}.npy"), head_mask.detach().cpu().numpy())
        np.savetxt(os.path.join(output_dir, f"head_importance_{i}.npy"), head_importance.detach().cpu().numpy())

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
            model, eval_dataloader, device, local_rank, output_dir, head_mask=new_head_mask
        )
        preds = np.argmax(preds, axis=1) if output_mode == "classification" else np.squeeze(preds)
        current_score = glue_compute_metrics(task, preds, labels)[metric_name]
        logger.info(
            "Masking: current score: %f, remaning heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )

    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.savetxt(os.path.join(output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask


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
    score_masking = glue_compute_metrics(task, preds, labels)[metric_name]
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = {}
    for layer in range(len(head_mask)):
        heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
        heads_to_prune[layer] = heads_to_mask
    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    logger.info(f"Heads to prune: {heads_to_prune}")
    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        model, eval_dataloader, device, local_rank, output_dir, compute_importance=False, head_mask=None
    )
    preds = np.argmax(preds, axis=1) if output_mode == "classification" else np.squeeze(preds)
    score_pruning = glue_compute_metrics(task, preds, labels)[metric_name]
    new_time = datetime.now() - before_time

    # calculate sparsity after pruning
    sparsity = 1 - pruned_num_params / original_num_params

    logger.info(
        "Pruning: original num of params: %.2e, after pruning %.2e, sparsity: %.2f",
        original_num_params,
        pruned_num_params,
        sparsity,
    )
    logger.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    logger.info("Pruning: speed ratio (new timing / original timing): %f percents", original_time / new_time * 100)

    return sparsity


def create_examples(lines):
    """
    Creates examples for dev set.
    Based on _create_examples class method of stsb and mnli processor
    """
    examples = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        guid = f"dev-{line[-1]}"
        text_a = line[0]
        text_b = line[1]
        label = line[2]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def load_examples(task, tokenizer, data_dir):
    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from dataset file
    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    if task == "mnli":
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]

    with open(f'{data_dir}/dev.tsv', "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter="\t"))
        examples = (
            create_examples(data)
        )

    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=512,
        output_mode=output_mode,
        # pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        # pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        # pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([0 for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


# Compute evaluation metrics for various GLUE tasks
def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli_two":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli_two_half":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans_mnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
