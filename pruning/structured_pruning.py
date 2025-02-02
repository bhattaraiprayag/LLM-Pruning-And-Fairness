# Based on https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py

import logging
import os
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset  # Subset needed if you uncomment line when preparing the dataset
from pruning.utils import load_examples, get_seed, mask_heads
from pruning.utils import get_device
from pruning.sparsity_check import structured_sparsity

logger = logging.getLogger(__name__)
logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True


def structured_pruning(model, tokenizer, seed, task, device, masking_threshold, exp_id, model_no):
    # Setup devices and distributed training
    local_rank = device
    device = get_device()
    n_gpu = 1
    # torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

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
    # val_data = Subset(val_data, list(range(min(subset_size, len(val_data)))))
    eval_sampler = SequentialSampler(val_data) # if local_rank == -1 else DistributedSampler(val_data)
    eval_dataloader = DataLoader(val_data, sampler=eval_sampler, batch_size=1)

    # set output directory
    output_dir = f'results/run{exp_id}/s-pruning'
    os.makedirs(output_dir, exist_ok=True)

    # set masking amount to 0.1 (default of original code)
    masking_amount = 0.1

    # perform masking
    head_mask = mask_heads(model, eval_dataloader, device, local_rank, output_dir, task, masking_amount, masking_threshold)
    # convert head_mask to tensor which is needed for applying the head mask to the model in the evaluation functions
    head_mask_tensor = torch.tensor(head_mask, dtype=torch.float32)
    
    # get final sparsity of the model
    sparsity = structured_sparsity(model, head_mask)
    print(f'Final sparsity: {sparsity}')

    return sparsity, head_mask_tensor

