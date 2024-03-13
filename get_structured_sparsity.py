import numpy as np

from transformers import (
    RobertaForSequenceClassification,
    HfArgumentParser,
)

from dataclasses import dataclass, field
from pruning.sparsity_check import structured_sparsity
from report.utils import run_info


@dataclass
class ExperimentArguments:
    run_nos: list = field(
        metadata={"help": "Specify list of run numbers."}
    )


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

    head_mask = np.loadtxt(f"results/run{run_no}/head_mask.npy")

    return structured_sparsity(model, head_mask)


def all_checks(run_no_list):
    output = {}
    for i in run_no_list:
        output[i] = single_check(i)

    return output


def main():
    parser = HfArgumentParser(ExperimentArguments)
    exp_args = parser.parse_args_into_dataclasses()[0]

    print(all_checks(exp_args.run_nos))


if __name__ == '__main__':
    main()
