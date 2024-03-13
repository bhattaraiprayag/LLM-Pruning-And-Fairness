import numpy as np
import transformers
from pruning.sparsity_check import structured_sparsity

def single_check(run_no):
    model = transformers.RobertaForSequenceClassification.from_pretrained(f"results/run{run_no}/pruned_model/",
                                                                          use_safetensors=True, local_files_only=True)
    head_mask = np.loadtxt(f"results/run{run_no}/s-pruning/head_mask.npy")

    return structured_sparsity(model, head_mask)

def all_checks(run_no_list):
    output = {}
    for i in run_no_list:
        output[i] = single_check(i)

    return output
