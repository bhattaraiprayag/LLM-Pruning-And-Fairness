# Based on code from https://github.com/sai-prasanna/bert-experiments/tree/master

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


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin"
}
def get_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

#def get_seed(args):
    #if args.seed is not None:
        #seed = int(args.seed)
       # random.seed(seed)
        #np.random.seed(seed)


#def get_seed(seed):
    #if seed is not None:
        #if isinstance(seed, int):
            #return seed
        #elif isinstance(seed, float):
           # return int(seed)
        #elif isinstance(seed, str):
            #try:
                #return int(seed)
            #except ValueError:
                #return seed
        #elif isinstance(seed, bytes) or isinstance(seed, bytearray):
            #return seed.decode("utf-8")

    #random.seed(None)
    #np.random.seed(None)
    #torch.manual_seed(None)
    #torch.cuda.manual_seed_all(None)


class ImpactTracker:
    def __init__(self, metrics, threshold):
        self.metrics = metrics
        self.threshold = threshold

    def track_impact(self, model):
        before_metrics = {metric: model.evaluate(metric) for metric in self.metrics}
        model.prune()  # Perform pruning or modifications
        after_metrics = {metric: model.evaluate(metric) for metric in self.metrics}

        for metric, change in zip(self.metrics, after_metrics - before_metrics):
            if abs(change) > self.threshold:
                print(f"Significant change detected for {metric}: {change}")

    def launch_impact_monitor(self, model):
        # Collect metrics before pruning
        before_metrics = {metric: model.evaluate(metric) for metric in self.metrics}

        # Perform pruning or modifications
        model.prune()

        # Collect metrics after pruning
        after_metrics = {metric: model.evaluate(metric) for metric in self.metrics}

        # Analyze and alert for significant changes
        for metric, change in zip(self.metrics, after_metrics - before_metrics):
            if abs(change) > self.threshold:
                print(f"Significant change detected for {metric}: {change}")


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


def load_tf_weights_in_roberta(model, config, tf_checkpoint_path):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Loading a TensorFlow model in PyTorch requires TensorFlow to be installed. Please see "
                          "https://www.tensorflow.org/install/ for installation instructions.")

    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))

    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")

        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            print("Skipping {}".format("/".join(name)))
            continue

        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue

            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise

        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    return model


class RobertaPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = RobertaConfig  # Update with the correct config class for RoBERTa
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP  # Update with the correct map
    load_tf_weights = load_tf_weights_in_roberta  # Implement a function to load RoBERTa TF weights
    base_model_prefix = "roberta"  # Update with the correct prefix for RoBERTa


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RobertaLayerNorm):  # Update with the correct LayerNorm class for RoBERTa
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

#Logger Setup
logger = logging.getLogger(__name__)

# This caching mechanism helps improve performance by avoiding unnecessary data loading for each epoch.
#def load_and_cache_examples(args, task, tokenizer, evaluate=False):

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            #pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            #pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            #pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

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
