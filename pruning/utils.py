import os
import logging
import re
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from torch import threshold
from transformers import RobertaModel, RobertaConfig
from transformers import glue_processors as processors
from torch.utils.data import TensorDataset
from transformers.data.metrics import simple_accuracy, acc_and_f1, pearson_and_spearman
from transformers.modeling_utils import PreTrainedModel
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin"
    }


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


def load_examples(task, tokenizer, data_dir):
    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from dataset file
    logger.info("Creating features from dataset file at %s", data_dir)
    label_list = processor.get_labels()
    if task == "mnli":
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]
    examples = (
        processor.get_dev_examples(data_dir)
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
