# Code amended from experiments/seat.py in https://github.com/McGill-NLP/bias-bench

import json
import os

from evaluation.utils.seat import SEATRunner, aggregate_results


def seatandweat(model, head_mask, tokenizer, exp_id, seed):
    runner = SEATRunner(
        experiment_id=exp_id,
        data_dir='evaluation/data/seat',
        n_samples=1000,
        parametric=False,
        model=model,
        head_mask=head_mask,
        tokenizer=tokenizer,
        seed=seed
    )
    all_results = runner()

    # aggregate results by bias (returns average absolute effect sizes for every type of bias)
    avg_es = aggregate_results(all_results)

    os.makedirs(f'results/run{exp_id}', exist_ok=True)  # maybe save as csv instead?
    with open(f'results/run{exp_id}/seatandweat_raw.json', 'w') as file:
        json.dump(all_results, file)
    with open(f'results/run{exp_id}/seatandweat_aggregated.json', 'w') as file:
        json.dump(avg_es, file)

    # only return gender bias values in dict
    return {k: avg_es[k] for k in ('SEAT_gender', 'WEAT_gender')}
