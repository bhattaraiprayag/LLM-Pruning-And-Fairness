import random
import numpy as np
import torch


def set_seed(seed):
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