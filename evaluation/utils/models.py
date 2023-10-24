# A portion from models.py and utils/experiment_id.py in https://github.com/McGill-NLP/bias-bench

# models.py
from functools import partial

import torch
import transformers

class RobertaModel:
    def __new__(self, model_name_or_path):
        return transformers.RobertaModel.from_pretrained(model_name_or_path))

# utils/experiment_id.py
def generate_experiment_id(
    name,
    model=None,
    model_name_or_path=None,
    bias_type=None,
    seed=None,
):
    experiment_id = f"{name}"

    # Build the experiment ID.
    if isinstance(model, str):
        experiment_id += f"_m-{model}"
    if isinstance(model_name_or_path, str):
        experiment_id += f"_c-{model_name_or_path}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id
