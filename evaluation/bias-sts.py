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
        scores_men = [i[0]['score'] for i in pred]
        test_df['scores'] = scores_men
        test_df.to_csv('men_new.csv', index=False)
        print('men done')

    with open(f'{thisdir}/evaluation/data/bias_sts/women.csv', mode='r') as csv_file:
        test_df = pd.read_csv(csv_file)
        # Get all sentence pairs in a list of tuples
        pair_list = test_df.to_records(index=False).tolist()
        # Make predictions with model
        pred = model_pipe(pair_list)
        scores_women = np.array([i[0]['score'] for i in pred])
        test_df['scores'] = scores_women
        test_df.to_csv('women_new.csv', index=False)
        print('women done')

    result = {'avg_abs_diff': np.sum(np.absolute(np.subtract(np.array(scores_women), np.array(scores_men)))) / len(scores_men)}

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

avg_abs_diff = bias_sts(pipe, '..')
print(avg_abs_diff)