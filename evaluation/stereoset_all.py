# Adapted from experiments/stereoset.py in https://github.com/McGill-NLP/bias-bench

import argparse
import json
import os

import transformers

from utils.stereoset import StereoSetRunner
import utils.models as models
from utils.experiment_id import generate_experiment_id
from utils.model_utils import _is_generative

def stereoset(model, tokenizer, exp_id):

    runner = StereoSetRunner(
            intrasentence_model=model,
            tokenizer=tokenizer,
            input_file=f"evaluation/data/stereoset/test.json",
            model_name_or_path='roberta',
            batch_size=1,
            is_generative=False
        )
    results = runner()

    # Save intermediate result
    os.makedirs(f'results/{exp_id}', exist_ok=True)
    with open(f"results/{exp_id}/stereoset.json", "w") as f:
        json.dump(results, f, indent=2)