# imports
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

# dataclass that contains all arguments needed to run the experiment
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

    seed: int = field(
        metadata={"help": "Specify random seed."}
    )

    task: str = field(
        metadata={"help": "Specify task. Options: 'mnli' or 'stsb'"},
    )

    pruning_method: Optional[str] = field(
        default='None',  # None means that the base models are evaluated without doing pruning
        metadata={"help": "Specify pruning method. Options: 'l1-unstructured', 'l1-unstructured-linear', 'l1-unstructured-invert', or None for no pruning."},  # add all options
    )

    device: int = field(
        default=0,
        metadata={"help": "Specify device that should be used. GPU: 0 (default), CPU: -1"},
    )

    model_no: int = field(
        default=1,
        metadata={"help": "Specify which model is used. The different models were fine-tuned on different splits of the datasets. Default: 1"},
    )


# main function that runs the experiment pipeline (evaluation and pruning dependent on arguments)
def main():
    parser = HfArgumentParser(ExperimentArguments)
    exp_args = parser.parse_args_into_dataclasses()[0]

    output_dir = 'results/performance'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # select model path based on task and model_no
    if exp_args.task == 'mnli':
        model_path = f'training/final_models/MNLI/model_no{exp_args.model_no}/'
    elif exp_args.task == 'stsb':
        model_path = f'training/final_models/STS-B/model_no{exp_args.model_no}/'
    else:
        raise ValueError(f'No model found for task {exp_args.task}')

    # set experiment seed
    get_seed(exp_args.seed)

    sparsity = [x/100 for x in range(0,100,5)]
    performance = {}

    # Pruning
    for i in sparsity:
        # load model
        model = RobertaForSequenceClassification.from_pretrained(
            model_path,
            use_safetensors=True,
            local_files_only=True
        )
        model.to(get_device())

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # prune
        if i!=0:
            pruner = MagnitudePrunerOneShot(model, exp_args.seed, exp_args.pruning_method, i)
            pruner.prune()

        # evaluate model "performance" (not fairness)
        eval_datasets = load_test_dataset(exp_args.task, exp_args.model_no)
        res_performance = evaluate_metrics(model, None, tokenizer, exp_args.task, eval_datasets)
        performance[i] = res_performance

    if exp_args.pruning_method=='random-unstructured':
        filename = f'{exp_args.task}_{exp_args.model_no}_{exp_args.pruning_method}_{exp_args.seed}'
    else:
        filename = f'{exp_args.task}_{exp_args.model_no}_{exp_args.pruning_method}'

    pd.DataFrame.from_dict(performance, orient='index').to_csv(f'{output_dir}/{filename}.csv')

if __name__ == '__main__':
    main()