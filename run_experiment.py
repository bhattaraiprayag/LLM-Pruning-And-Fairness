# imports
import pandas as pd
import os

import transformers
from transformers import (
    # AutoConfig,
    # AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaForSequenceClassification,
    TextClassificationPipeline,
    # DataCollatorWithPadding,
    # EvalPrediction,
    HfArgumentParser,
    # PretrainedConfig,
    # Trainer,
    # TrainingArguments,
    # default_data_collator,
    set_seed,
)

from dataclasses import dataclass, field
from typing import Optional


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
        metadata={"help": "MNLI or STS-B"},
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
    exp_args = parser.parse_args_into_dataclasses()

    # specify current directory
    thisdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # load dataframe that stores the results (every run adds a new row)
    results_df = pd.read_csv(f'{thisdir}/results/results.csv')

    # determine ID of this run
    id = results_df['ID'].max() + 1

    # create output/results folder directory (one folder per run) -> used to store bigger/more detailed outputs
    outdir = os.path.join(thisdir, f'/results/run{str(id)}')
    os.mkdir(outdir)

    # select model path based on task
    if exp_args.task == 'MNLI':
        model_path = f'{thisdir}/final_models/MNLI/'
    elif exp_args.task == 'STS-B':
        model_path = f'{thisdir}/final_models/STS-B/'
    else:
        raise ValueError(f'No model found for task {exp_args.task}')

    # load model
    model = RobertaForSequenceClassification.from_pretrained(
        model_path,
        use_safetensors=True,
        local_files_only=True
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(exp_args.model_path)

    # create pipeline
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, max_length=512, truncation=True, padding=True)

    # set seed before running the experiment (??? needed if we put it directly into pruning functions ???)
    set_seed(exp_args.seed)

    # pruning (skipped if pruning == None)
    # ideally set up one pruning function

    # model evaluation

    # fairness evaluation
    # ideally: set up one evaluation function

    # store everything in data frame (code still missing to create results_run)
    # results_df = pd.concat([results_df, results_run])


if __name__ == '__main__':
    main()
