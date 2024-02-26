# imports
from transformers import (
    RobertaForSequenceClassification,
    HfArgumentParser,
)

from dataclasses import dataclass, field, asdict
from pruning.utils import get_device, get_seed

from pruning.utils import check_sparsity

# dataclass that contains all arguments needed to run the experiment
@dataclass
class ExperimentArguments:

    task: str = field(
        metadata={"help": "Specify task. Options: 'mnli' or 'stsb'"},
    )

    model_no: int = field(
        default=1,
        metadata={"help": "Specify which model is used. The different models were fine-tuned on different splits of the datasets. Default: 1"},
    )

def main():
    parser = HfArgumentParser(ExperimentArguments)
    exp_args = parser.parse_args_into_dataclasses()[0]

    # select model path based on task and model_no
    if exp_args.task == 'mnli':
        model_path = f'training/final_models/MNLI/model_no{exp_args.model_no}/'
    elif exp_args.task == 'stsb':
        model_path = f'training/final_models/STS-B/model_no{exp_args.model_no}/'
    else:
        raise ValueError(f'No model found for task {exp_args.task}')

    # load model
    model = RobertaForSequenceClassification.from_pretrained(
        model_path,
        use_safetensors=True,
        local_files_only=True
    )
    model.to(get_device())

    print(check_sparsity(model))

if __name__ == '__main__':
    main()