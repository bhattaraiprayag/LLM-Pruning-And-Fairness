import csv
import numpy as np

import pandas as pd
import os

import transformers
from transformers import (
    # AutoConfig,
    # AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaForSequenceClassification,
    TextClassificationPipeline,
    # DataCollatorWithPadding,
    # EvalPrediction,
    HfArgumentParser,
    # PretrainedConfig,
    # Trainer,
    # TrainingArguments,
    # default_data_collator,
    set_seed,
)


def bias_sts(model_pipe, thisdir):
    # Input: model_pipe with the transformers pipeline containing the model and tokenizer
    # Output: results dictionary with the average absolute difference between the similarity scores of male and female sentence pairs

    # Load csv with test sentences, then make model predictions for each
    with open(f'{thisdir}/evaluation/data/bias_sts/men.csv', mode='r') as csv_file:
        test_df = pd.read_csv(csv_file)
        # Get all sentence pairs in a list of tuples
        pair_list = test_df.to_records(index=False).tolist()
        # Make predictions with model
        pred = model_pipe(pair_list)
        scores_men = np.array([i['score'] for i in pred])
        print('men done')

    with open(f'{thisdir}/evaluation/data/bias_sts/women.csv', mode='r') as csv_file:
        test_df = pd.read_csv(csv_file)
        # Get all sentence pairs in a list of tuples
        pair_list = test_df.to_records().tolist()
        # Make predictions with model
        pred = model_pipe(pair_list)
        scores_women = np.array([i['score'] for i in pred])
        print('women done')

    result = {'avg_abs_diff': np.sum(np.absolute(np.subtract(scores_women, scores_men))) / len(scores_men)}

    return result


model = RobertaForSequenceClassification.from_pretrained(
    '../training/final_models/STS-B',
    use_safetensors=True,
    local_files_only=True
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('../training/final_models/STS-B')

# create pipeline
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, max_length=512, truncation=True,
                                  padding=True)

pred = bias_sts(pipe, '..')
print(pred)