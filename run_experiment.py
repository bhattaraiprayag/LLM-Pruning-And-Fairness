# imports
import pandas as pd

from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    TextClassificationPipeline,
    HfArgumentParser,
    set_seed,
)

from dataclasses import dataclass, field
from typing import Optional

from evaluation.seat import seatandweat


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
        metadata={"help": "Specify pruning method (...)"},  # add all options
    )

    sparsity_level: Optional[float] = field(
        default=None,
        metadata={"help": "Specify pruning method (None, ... )"},  # add all options
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

    # load model
    model = RobertaForSequenceClassification.from_pretrained(
        model_path,
        use_safetensors=True,
        local_files_only=True
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # create pipeline
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, max_length=512, truncation=True, padding=True)

    # set seed before running the experiment (??? needed if we put it directly into pruning functions ???)
    set_seed(exp_args.seed)

    # pruning (skipped if pruning == None)
    # ideally set up one pruning function

    # model evaluation

    # fairness evaluation
    # ideally: set up one evaluation function
    seatandweat(model, tokenizer, id, exp_args.seed) # only print results so far and does not return anything!

    # store everything in data frame (code still missing to create results_run)
    # results_df = pd.concat([results_df, results_run])


if __name__ == '__main__':
    main()
