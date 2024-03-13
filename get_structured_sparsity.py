import numpy as np
import os
import pandas as pd
from datetime import date

from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    TextClassificationPipeline,
    HfArgumentParser,
    set_seed,
)

from dataclasses import dataclass, field, asdict
from typing import Optional
from pruning.utils import get_seed, get_device
from pruning.magnitude_pruner import MagnitudePrunerOneShot
from evaluation.performance import load_test_dataset, evaluate_metrics
from pruning.sparsity_check import structured_sparsity

@dataclass
class ExperimentArguments:
    """
    Arguments needed to run the experiment
    - seed
    - task
    - pruning_method
    - sparsity_level
    - device
    - temperature
    ...
    """

    run_nos: list = field(
        metadata={"help": "Specify list of run numbers."}
    )

def single_check(run_no):
    model = RobertaForSequenceClassification.from_pretrained(f"results/run{run_no}/pruned_model/",
                                                                          use_safetensors=True, local_files_only=True)
    head_mask = np.loadtxt(f"results/run{run_no}/s-pruning/head_mask.npy")

    return structured_sparsity(model, head_mask)

def all_checks(run_no_list):
    output = {}
    for i in run_no_list:
        output[i] = single_check(i)

    return output

def main():
    print(all_checks(run_nos))

if __name__ == '__main__':
    main()