# imports
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
from pruning.utils import set_seed
from pruning.magnitude_pruner import MagnitudePrunerOneShot
from evaluation.performance import load_eval_dataset, evaluate_metrics
from evaluation.seat import seatandweat
from evaluation.stereoset import stereoset
from evaluation.bias_nli import bias_nli


# dataclass that contains all arguments needed to run the experiment
@dataclass
class ExperimentArguments:
    """
    Arguments needed to run the experiment
    - seed
    - task
    - pruning_method
    - sparsity_level
    ...
    """

    seed: int = field(
        metadata={"help": "random seed"}
    )

    task: str = field(
        metadata={"help": "mnli or stsb"},
    )

    pruning_method: Optional[str] = field(
        default=None,  # None means that the base models are evaluated without doing pruning
        metadata={"help": "Specify pruning method. Options: 'l1-unstructured', 'l1-unstructured-linear', 'l1-unstructured-invert', or None for no pruning."},  # add all options
    )

    sparsity_level: Optional[float] = field(
        default=None,
        metadata={"help": "Specify desired sparsity level. From 0 to 1.)"},  # add all options
    )

    device: int = field(
        default=0,
        metadata={"help": "Specify device that should be used. GPU: 0 (default), CPU: -1"},
    )


# main function that runs the experiment pipeline (evaluation and pruning dependent on arguments)
def main():
    parser = HfArgumentParser(ExperimentArguments)
    exp_args = parser.parse_args_into_dataclasses()[0]

    # load dataframe that stores the results (every run adds a new row)
    results_df = pd.read_csv('results/results.csv')

    # determine ID of this run
    if results_df.empty:
        id = 1
    else:
        id = results_df['ID'].max() + 1

    # NOT NEEDED?? create output/results folder directory (one folder per run) to put into functions
    # outdir = f'/results/run{str(id)}'
    # os.mkdir(outdir)

    # select model path based on task
    if exp_args.task == 'mnli':
        model_path = 'training/final_models/MNLI/'
    elif exp_args.task == 'stsb':
        model_path = 'training/final_models/STS-B/'
    else:
        raise ValueError(f'No model found for task {exp_args.task}')
    
    # set experiment seed
    set_seed(exp_args.seed)

    # load model
    model = RobertaForSequenceClassification.from_pretrained(
        model_path,
        use_safetensors=True,
        local_files_only=True
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # create pipeline
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, max_length=512, truncation=True, padding=True, device=exp_args.device)

    # pruning (skipped if pruning == None)
    if exp_args.pruning_method is not None:
        pruner = MagnitudePrunerOneShot(model, exp_args.seed, exp_args.pruning_method, exp_args.sparsity_level)
        pruner.prune()

    # evaluate model "performance" (not fairness)
    eval_datasets = load_eval_dataset(exp_args.task)
    res_performance = evaluate_metrics(model, tokenizer, exp_args.task, eval_datasets, id)
    print(res_performance)

    # fairness evaluation
    # ideally: set up one evaluation function
    res_seatandweat = seatandweat(model, tokenizer, id, exp_args.seed)
    res_stereoset = stereoset(model, tokenizer, id)
    res_bnli = bias_nli(pipe, id)

    results_run = {**asdict(exp_args), **res_performance, **res_seatandweat, **res_stereoset, **res_bnli}
    results_run.update({'ID': id, 'date': date.today()})
    print(results_run)

    # store everything in data frame
    results_df = pd.concat([results_df, pd.DataFrame.from_dict([results_run])], ignore_index=True)
    # save updated csv file
    results_df.to_csv('results/results.csv', index=False)


if __name__ == '__main__':
    # Sample run: python run_experiment.py --task stsb --pruning_method l1-unstructured --sparsity_level 0.5 --seed 42
    main()