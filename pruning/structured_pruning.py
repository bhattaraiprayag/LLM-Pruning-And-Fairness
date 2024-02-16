# Based on https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py

import logging
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from utils import load_examples, get_seed, mask_heads, prune_heads
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from utils import get_device

logger = logging.getLogger(__name__)
logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True


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

    # Prepare dataset
    data_dir =
    val_data = load_examples(task, tokenizer, data_dir)
    # use subset of data if needed for debugging
    # subset_size = 100
    # eval_data = Subset(val_data, list(range(min(subset_size, len(val_data)))))
    eval_sampler = SequentialSampler(val_data) if local_rank == -1 else DistributedSampler(val_data)
    eval_dataloader = DataLoader(val_data, sampler=eval_sampler, batch_size=1)

    # set output directory
    output_dir =

    # perform pruning
    head_mask = mask_heads(model, eval_dataloader, device, local_rank, output_dir, task, masking_amount, masking_threshold)
    prune_heads(model, eval_dataloader, device, local_rank, output_dir, task, head_mask)