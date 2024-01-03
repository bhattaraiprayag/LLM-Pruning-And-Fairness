# Function for the bias_nli evaluation
# To note: templates must be available at the correct filepath

import os
import csv

def bias_nli(model_pipe):
    # Input: model_pipe with the transformers pipeline containing the model and tokenizer
    # Output: results dictionary with net neutral and fraction neutral scores

    thisdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Load csv with test sentences, then make model predictions for each
    with open(f'{thisdir}/evaluation/data/bias_nli/occupation_gender.csv', mode='r') as csv_file:
        test_dict = csv.DictReader(csv_file)
        pair_list = []
        # Get all premise/hypothesis pairs in a list
        for row in test_dict:
            pair_list.append((row["premise"],row["hypothesis"]))
        # Make predictions with model
        prediction = pipe(pair_list)
        results = {}
        results['net_neutral'] = sum([next((item.get('score') for item in i if item["label"] == "neutral"), False) for i in prediction])/len(pair_list)
        results['fraction_neutral'] = sum([1 for i in prediction if max(i, key=lambda x:x['score'])['label']=='neutral'])/len(pair_list)

    return results