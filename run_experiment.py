# imports
import transformers
from transformers import (
    #AutoConfig,
    #AutoModelForSequenceClassification,
    #AutoTokenizer,
    #DataCollatorWithPadding,
    #EvalPrediction,
    HfArgumentParser,
    #PretrainedConfig,
    #Trainer,
    #TrainingArguments,
    #default_data_collator,
    #set_seed,
)

from dataclasses import dataclass, field


# dataclass that contains all arguments needed to run the experiment
@dataclass
class ExperimentArguments:
    """
    Arguments needed to run the experiment
    - model_path (maybe replace the path by just specifying MNLI or STS-B and then select the path (only works if we only have these two options))

    missing:
    - selected pruning method
    - output directory (include check to avoid that anything gets overwritten!)
    ...
    """
    model_path: str = field(
        metadata={"help": "Path to fine-tuned model"}
    )

# main function that runs the pipeline (evaluations and pruning dependent on arguments)
def main():
    parser = HfArgumentParser(ExperimentArguments)
    exp_args = parser.parse_args_into_dataclasses()

    # load model
    model = transformers.RobertaForSequenceClassification.from_pretrained(
        exp_args.model_path,
        use_safetensors=True,
        local_files_only=True
    )


    # evaluation 1


    # pruning


    # evaluation 2


if __name__ == "__main__":
    main()
