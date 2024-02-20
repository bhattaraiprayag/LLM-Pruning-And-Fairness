# Based on https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py

import logging
import os
from torch.utils.data import DataLoader, SequentialSampler, Subset  # Subset needed if you uncomment line when preparing the dataset
from torch.utils.data.distributed import DistributedSampler
from pruning.utils import load_examples, get_seed, mask_heads, prune_heads, check_sparsity
from pruning.utils import get_device

logger = logging.getLogger(__name__)
logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True


def structured_pruning(model, tokenizer, seed, task, device, masking_amount, masking_threshold, exp_id, model_no):
    # Setup devices and distributed training
    local_rank = device
    device = get_device()
    n_gpu = 1
    #torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(device, n_gpu, bool(local_rank != -1)))

    # Set seed
    get_seed(seed)

    # change task name stsb to sts-b (necessary for predefined functions)
    if task == "stsb":
        task = "sts-b"

    # Prepare dataset
    data_dir = f'training/glue_data/{task}/model_no{model_no}'
    val_data = load_examples(task, tokenizer, data_dir)
    # use subset of data if needed for debugging
    # subset_size = 100
    # eval_data = Subset(val_data, list(range(min(subset_size, len(val_data)))))
    eval_sampler = SequentialSampler(val_data) if local_rank == -1 else DistributedSampler(val_data)
    eval_dataloader = DataLoader(val_data, sampler=eval_sampler, batch_size=1)

    # set output directory
    output_dir = f'results/run{exp_id}/s-pruning'
    os.makedirs(output_dir, exist_ok=True)

    # perform pruning
    head_mask = mask_heads(model, eval_dataloader, device, local_rank, output_dir, task, masking_amount, masking_threshold)
    prune_heads(model, eval_dataloader, device, local_rank, output_dir, task, head_mask)

    # return the final sparsity of the model
    return check_sparsity(model)

