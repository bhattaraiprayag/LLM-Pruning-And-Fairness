# Code amended from experiments/seat.py in https://github.com/McGill-NLP/bias-bench

import json
import os

from evaluation.utils.seat import SEATRunner
#import utils.models as models
#from utils.experiment_id import generate_experiment_id


def seatandweat(model, tokenizer, exp_id, seed):
    runner = SEATRunner(
        experiment_id=exp_id,
        data_dir='evaluation/data/seat',
        n_samples=1000,
        parametric=False,
        model=model,
        tokenizer=tokenizer,
        seed=seed
    )
    results = runner()
    print(results)

    # TO DO: return most important results (which are the most important?)

    os.makedirs(f'results/{exp_id}', exist_ok=True) # maybe save as csv instead?
    with open(f'results/{exp_id}/seatandweat.json', 'w') as file:
        json.dump(results, file)
