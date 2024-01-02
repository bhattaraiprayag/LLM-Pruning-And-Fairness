# imports
import pandas as pd
import transformers
from transformers import (
    # AutoConfig,
    # AutoModelForSequenceClassification,
    # AutoTokenizer,
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
    - id
    - seed
    - model_path (maybe select the path automatically based on task)
    - task
    - pruning_method
    - sparsity_level
    ...
    """
    id: int = field(
        metadata={"help": "ID of experiment run"}
    )

    seed: int = field(
        metadata={"help": "random seed"}
    )

    model_path: str = field(
        metadata={"help": "Path to fine-tuned model"}
    )

    task: str = field(
        metadata={"help": "MNLI or STS-B"},
    )

    pruning_method: Optional[str] = field(
        default=None, # None means that the base models are evaluated without doing pruning
        metadata={"help": "Specify pruning method (...)"}, # add all options
    )

    sparsity_level: Optional[float] = field(
        default=None,
        metadata={"help": "Specify pruning method (None, ... )"},  # add all options
    )

# main function that runs the pipeline (evaluation and pruning dependent on arguments)
def main():
    parser = HfArgumentParser(ExperimentArguments)
    exp_args = parser.parse_args_into_dataclasses()

    # load dataframe that stores the results (every run adds a new row)
    results_df = pd.read_csv("results/results.csv")

    # MISSING CHANGE: create ID instead of setting it manually
    # check if ID already exists in data frame, if yes throw error
    if exp_args.id in results_df['ID'].values:
        raise ValueError("Experiment ID already exists.")

    # specify current directory

    # create output/results folder directory (one folder per run) -> used to store bigger/more detailed outputs

    # set seed before running the experiment (??? needed if we put it directly into pruning functions ???)
    set_seed(exp_args.seed)

    # load model
    model = transformers.RobertaForSequenceClassification.from_pretrained(
        exp_args.model_path,
        use_safetensors=True,
        local_files_only=True
    )

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(exp_args.model_path)

    # pruning (skipped if pruning == None)
    # ideally set up one pruning function

    # model evaluation

    # fairness evaluation
    # ideally: set up one evaluation function

    # store everything in data frame (code still missing to create results_run)
    #results_df = pd.concat([results_df, results_run])

if __name__ == "__main__":
    main()
