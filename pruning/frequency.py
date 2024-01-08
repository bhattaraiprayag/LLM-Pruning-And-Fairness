from transformers import glue_processors
from transformers import glue_compute_metrics
from collections import Counter
from scipy.stats import norm
import numpy as np
import json
import subprocess
np.random.seed(1337)

# create a baseline prediction strategy based on the frequency of labels in the training data and evaluates
# the performance of this baseline on the development set for each task in the GLUE benchmark dataset.

counter_dict = {}
for task in ["CoLA", "MNLI", "MNLI-mm", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B", "WNLI"]:
    processor = glue_processors[task.lower()]()
    data_dir = "MNLI" if task.startswith("MNLI") else task
    eval_examples = processor.get_dev_examples(f"../data/glue/{data_dir}")
    if task == "STS-B":
        items = []
        for example in eval_examples:
            items.append(float(example.label))
        counter_dict[task] = norm.fit(items)
    else:
        counter = Counter()
        for example in eval_examples:
            counter[example.label] += 1
        counter_dict[task] = counter

counter_dict

### Frequency baseline prediction

freq_base_line_prediction = {}
for task in counter_dict.keys():
    if task != "STS-B":
        freq_base_line_prediction[task] = counter_dict[task].most_common()[0][0]
    else:
        freq_base_line_prediction[task] = counter_dict[task]

freq_base_line_prediction

### Frequency baseline prediction Benchmark

freq_baseline = {}
for task in freq_base_line_prediction.keys():
    processor = glue_processors[task.lower()]()
    data_dir = "MNLI" if task.startswith("MNLI") else task
    eval_examples = processor.get_dev_examples(f"../data/glue/{data_dir}")
    label_list = processor.get_labels()
    prediction = freq_base_line_prediction[task]
    if task == "STS-B":
        labels = []
        predictions = np.random.normal(prediction[0], prediction[1], len(eval_examples))
        for example in eval_examples:
            labels.append(float(example.label))
        labels = np.array(labels)
    else:
        labels = []
        predictions = []
        for example in eval_examples:
            labels.append(label_list.index(example.label))
            predictions.append(label_list.index(prediction))
        predictions = np.array(predictions)
        labels = np.array(labels)
    results = glue_compute_metrics(task.lower(), predictions, labels)
    if task.startswith("MNLI"):
        if "MNLI" not in freq_baseline:
            freq_baseline["MNLI"] = {}
        for k, v in results.items():
            freq_baseline["MNLI"][f"{task.lower()}_{k}"] = v
    else:
        freq_baseline[task] = results

freq_baseline


def write_results(results, output_file_path):
    with open(output_file_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

subprocess.run(["mkdir", "-p", "../experiments/freq_baseline"], check=True)

#!mkdir -p ../experiments/freq_baseline
write_results(freq_baseline, "../experiments/freq_baseline/results.json")