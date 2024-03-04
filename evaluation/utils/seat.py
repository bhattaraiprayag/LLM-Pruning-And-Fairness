# Adapted from https://github.com/McGill-NLP/bias-bench

import json
import os
import random
import re

import numpy as np
import torch

import evaluation.utils.weat as weat

from pruning.utils import get_device


class SEATRunner:
    """Runs SEAT tests for a given HuggingFace transformers model.

    Implementation taken from: https://github.com/W4ngatang/sent-bias.
    """

    # Extension for files containing SEAT tests.
    TEST_EXT = ".jsonl"

    def __init__(
            self,
            model,
            tokenizer,
            data_dir,
            experiment_id,
            head_mask=None,
            n_samples=100000,
            parametric=False,
            seed=0,
    ):
        """Initializes a SEAT test runner.

        Args:
            model: HuggingFace model (e.g., BertModel) to evaluate.
            tokenizer: HuggingFace tokenizer (e.g., BertTokenizer) used for pre-processing.
            data_dir (`str`): Path to directory containing the SEAT tests.
            experiment_id (`str`): Experiment identifier. Used for logging.
            head_mask: array that indicates attention heads that are to be masked
            n_samples (`int`): Number of permutation test samples used when estimating p-values
                (exact test is used if there are fewer than this many permutations).
            parametric (`bool`): Use parametric test (normal assumption) to compute p-values.
            seed (`int`): Random seed.
        """
        self._model = model
        self._head_mask = head_mask
        self._tokenizer = tokenizer
        self._data_dir = data_dir
        self._experiment_id = experiment_id
        self._n_samples = n_samples
        self._parametric = parametric
        self._seed = seed

    def __call__(self):
        """Runs specified SEAT tests.

        Returns:
            `list` of `dict`s containing the SEAT test results.
        """
        random.seed(self._seed)
        np.random.seed(self._seed)

        all_tests = sorted(
            [
                entry[: -len(self.TEST_EXT)]
                for entry in os.listdir(self._data_dir)
                if not entry.startswith(".") and entry.endswith(self.TEST_EXT)
            ],
            key=_test_sort_key,
        )

        # run all SEAT tests
        tests = all_tests

        results = []
        for test in tests:

            # Load the test data.
            encs = _load_json(os.path.join(self._data_dir, f"{test}{self.TEST_EXT}"))

            encs_targ1 = _encode(
                self._model, self._head_mask, self._tokenizer, encs["targ1"]["examples"]
            )
            encs_targ2 = _encode(
                self._model, self._head_mask, self._tokenizer, encs["targ2"]["examples"]
            )
            encs_attr1 = _encode(
                self._model, self._head_mask, self._tokenizer, encs["attr1"]["examples"]
            )
            encs_attr2 = _encode(
                self._model, self._head_mask, self._tokenizer, encs["attr2"]["examples"]
            )

            encs["targ1"]["encs"] = encs_targ1
            encs["targ2"]["encs"] = encs_targ2
            encs["attr1"]["encs"] = encs_attr1
            encs["attr2"]["encs"] = encs_attr2

            # Run the test on the encodings.
            esize, pval = weat.run_test(
                encs, n_samples=self._n_samples, parametric=self._parametric
            )

            results.append(
                {
                    "experiment_id": self._experiment_id,
                    "test": test,
                    "p_value": pval,
                    "effect_size": esize,
                }
            )

        return results


def _test_sort_key(test):
    """Return tuple to be used as a sort key for the specified test name.
    Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    """
    key = ()
    prev_end = 0
    for match in re.finditer(r"\d+", test):
        key = key + (test[prev_end: match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key


def _split_comma_and_check(arg_str, allowed_set, item_type):
    """Given a comma-separated string of items, split on commas and check if
    all items are in allowed_set -- item_type is just for the assert message.
    """
    items = arg_str.split(",")
    for item in items:
        if item not in allowed_set:
            raise ValueError(f"Unknown {item_type}: {item}!")
    return items


def _load_json(sent_file):
    """Load from json. We expect a certain format later, so do some post processing."""
    all_data = json.load(open(sent_file, "r"))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
        v["examples"] = examples

    return all_data


def _encode(model, head_mask, tokenizer, texts):
    encs = {}
    for text in texts:
        # Encode each example.
        inputs = tokenizer(text, return_tensors="pt")
        inputs = inputs.to(get_device())
        outputs = model(**inputs, output_hidden_states=True, head_mask=head_mask)

        # Average over the last layer of hidden representations.
        enc = outputs.hidden_states[-1]  # line wants to get last hidden state, should be the same as the last item in hidden_states (https://stackoverflow.com/questions/61323621/how-to-understand-hidden-states-of-the-returns-in-bertmodelhuggingface-transfo)
        enc = enc.mean(dim=1)

        # Following May et al., normalize the representation.
        encs[text] = enc.detach().view(-1).cpu().numpy()
        encs[text] /= np.linalg.norm(encs[text])

    return encs


# newly added functions
def compute_avg_effect_size(list):
    es = [d['effect_size'] for d in list]
    return sum(abs(e) for e in es) / len(es)


def aggregate_results(all_results):
    seat_gender_tests = ['sent-weat6',
                         'sent-weat6b',
                         'sent-weat7',
                         'sent-weat7b',
                         'sent-weat8',
                         'sent-weat8b',
                         'heilman_double_bind_competent_1+3-',
                         'heilman_double_bind_competent_1',
                         'heilman_double_bind_competent_1-',
                         'heilman_double_bind_competent_one_sentence',
                         'heilman_double_bind_competent_one_word',
                         'heilman_double_bind_likable_1+3-',
                         'heilman_double_bind_likable_1',
                         'heilman_double_bind_likable_1-',
                         'heilman_double_bind_likable_one_sentence',
                         'heilman_double_bind_likable_one_word',
                         'sent-heilman_double_bind_competent_one_word',
                         'sent-heilman_double_bind_likable_one_word',
                         ]
    seat_gender = [d for d in all_results if d.get('test') in seat_gender_tests]

    seat_race_tests = ['sent-weat3',
                       'sent-weat3b',
                       'sent-weat4',
                       'sent-weat5',
                       'sent-weat5b',
                       'angry_black_woman_stereotype',
                       'angry_black_woman_stereotype_b',
                       'sent-angry_black_woman_stereotype_b',
                       'sent-angry_black_woman_stereotype'
                       ]
    seat_race = [d for d in all_results if d.get('test') in seat_race_tests]

    seat_illness = [d for d in all_results if d.get('test') == 'sent-weat9']

    seat_religion_tests = ['sent-religion1',
                           'sent-religion1b',
                           'sent-religion2',
                           'sent-religion2b'
                           ]
    seat_religion = [d for d in all_results if d.get('test') in seat_religion_tests]

    weat_gender_tests = ['weat6',
                         'weat6b',
                         'weat7',
                         'weat7b',
                         'weat8',
                         'weat8b'
                         ]
    weat_gender = [d for d in all_results if d.get('test') in weat_gender_tests]

    weat_race_tests = ['weat3',
                       'weat3b',
                       'weat4',
                       'weat5',
                       'weat5b'
                       ]
    weat_race = [d for d in all_results if d.get('test') in weat_race_tests]

    weat_illness = [d for d in all_results if d.get('test') == 'weat9']

    all_avg_es = {'SEAT_gender': compute_avg_effect_size(seat_gender),
                  'SEAT_race': compute_avg_effect_size(seat_race),
                  'SEAT_illness': compute_avg_effect_size(seat_illness),
                  'SEAT_religion': compute_avg_effect_size(seat_religion),
                  'WEAT_gender': compute_avg_effect_size(weat_gender),
                  'WEAT_race': compute_avg_effect_size(weat_race),
                  'WEAT_illness': compute_avg_effect_size(weat_illness)
                  }

    return all_avg_es
