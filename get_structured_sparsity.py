import numpy as np

from transformers import (
    RobertaForSequenceClassification,
    HfArgumentParser,
)

from dataclasses import dataclass, field
from pruning.sparsity_check import structured_sparsity
from report.utils import run_info


def single_check(run_no):
    info = run_info(run_no)
    if info.get('task') == 'mnli':
        model = RobertaForSequenceClassification.from_pretrained(
            f"training/final_models/MNLI/model_no{info.get('model_no')}",
            use_safetensors=True, local_files_only=True)
    elif info.get('task') == 'stsb':
        model = RobertaForSequenceClassification.from_pretrained(
            f"training/final_models/STS-B/model_no{info.get('model_no')}",
            use_safetensors=True, local_files_only=True)
    else:
        raise ValueError("task not supported")

    head_mask = np.loadtxt(f"results/head_masks/head_mask{run_no}.npy")

    return structured_sparsity(model, head_mask)


def all_checks(run_no_list):
    output = {}
    for i in run_no_list:
        output[i] = single_check(i)

    return output


def main():
    run_nos = [136, 137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205]
    print(all_checks(run_nos))


if __name__ == '__main__':
    main()
