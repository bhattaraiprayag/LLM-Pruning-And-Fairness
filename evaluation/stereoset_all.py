# Adapted from StereoSet processing in https://github.com/McGill-NLP/bias-bench

import argparse
from collections import Counter, OrderedDict, defaultdict
import glob
import json
import os
import re

import transformers
import numpy as np

from utils.stereoset import StereoSetRunner
import utils.models as models
from utils.experiment_id import generate_experiment_id
from utils.model_utils import _is_generative
from utils.stereoset import dataloader
import utils.stereoset_stats as ss


def stereoset(model, tokenizer, exp_id):

    runner = StereoSetRunner(
            intrasentence_model=model,
            tokenizer=tokenizer,
            input_file=f"evaluation/data/stereoset/test.json",
            model_name_or_path='roberta',
            batch_size=1,
            is_generative=False
        )
    individuals = runner()

    # Save intermediate result
    os.makedirs(f'results/{exp_id}', exist_ok=True)
    with open(f"results/{exp_id}/stereoset_raw.json", "w") as f:
        json.dump(individuals, f, indent=2)

    # Getting actual score
    overall = ss.parse_file(f"evaluation/data/stereoset/test.json", f"results/{exp_id}/stereoset_raw.json")

    with open(f"results/{exp_id}/stereoset_results.json", "w+") as f:
        json.dump(overall, f, indent=2)

    # Getting values to return
    results = {}
    results['ss_lm_gender'] = overall['intrasentence']['gender']['LM Score']
    results['ss_ss_gender'] = overall['intrasentence']['gender']['SS Score']