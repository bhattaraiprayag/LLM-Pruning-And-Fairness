# imports
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


# dataclass that contains all arguments needed to run the experiment
@dataclass
class ExperimentArguments:
    """
    Arguments needed to run the experiment
    - id
    - seed
    - model_path (maybe replace the path by just specifying MNLI or STS-B and then select the path (only works if we only have these two options))

    missing:
    - selected pruning method
    - output directory (include check to avoid that anything gets overwritten!)
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


# main function that runs the pipeline (evaluations and pruning dependent on arguments)
def main():
    parser = HfArgumentParser(ExperimentArguments)
    exp_args = parser.parse_args_into_dataclasses()

    # load data frame that stores the results (every run adds a new row)

    # check if ID already exists in data frame, if yes throw error

    # set seed before running the experiment
    set_seed(exp_args.seed)

    # load model
    model = transformers.RobertaForSequenceClassification.from_pretrained(
        exp_args.model_path,
        use_safetensors=True,
        local_files_only=True
    )

    # evaluation 1
    # set up one evaluation function that return all values in a dict

    # pruning

    # evaluation 2

    # store everything in data frame


if __name__ == "__main__":
    main()
