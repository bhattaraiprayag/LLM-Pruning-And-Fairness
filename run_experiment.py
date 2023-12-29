# imports
import transformers
from dataclasses import dataclass, field


# dataclass that contains all arguments needed to run the experiment
@dataclass
class ExperimentArguments:





# main function that runs the pipeline (evaluations and pruning dependent on arguments)
def main():
    # load model
    model = transformers.RobertaForSequenceClassification.from_pretrained(
        "models/MNLI_model/",
        use_safetensors=True,
        local_files_only=True
    )


    # evaluation 1


    # pruning


    # evaluation 2


if __name__ == "__main__":
    main()
