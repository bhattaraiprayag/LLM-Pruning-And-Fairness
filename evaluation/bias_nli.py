# Function for the bias_nli evaluation
# To note: templates must be available at the correct filepath

import csv
import os
from tqdm import tqdm

def bias_nli(model_pipe, exp_id):
    # Input: model_pipe with the transformers pipeline containing the model and tokenizer
    # Output: results dictionary with net neutral and fraction neutral scores, also saves csv with predictions

    # Load csv with test sentences, then make model predictions for each
    with open(f'evaluation/data/bias_nli/occupation_gender.csv', mode='r') as csv_file:
        test_dict = csv.DictReader(csv_file)
        pair_list = []
        word_list = []
        # Get all premise/hypothesis pairs in a list
        for row in tqdm(test_dict, desc="Loading bias-nli test sentences"):
            pair_list.append((row["premise"],row["hypothesis"]))
            word_list.append([row['premise_filler_word'], row['hypothesis_filler_word'], row["premise"],row["hypothesis"]])

    # Make predictions with model
    # prediction = model_pipe(pair_list)
    batch_size = 1000
    prediction = []
    for i in tqdm(range(0, len(pair_list), batch_size), desc="Making predictions"):
        prediction.extend(model_pipe(pair_list[i:i+batch_size]))

    # Save predictions
    os.makedirs(f'results/run{exp_id}', exist_ok=True)
    with open(f'results/run{exp_id}/bias_nli.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["premise_filler_word", "hypothesis_filler_word", "premise", "hypothesis", "neutral", "entailment", "contradiction"])
        for i, item in enumerate(prediction):
            line = word_list[i] + [[i['score'] for i in item if i['label']=='neutral'][0], [i['score'] for i in item if i['label']=='entailment'][0], [i['score'] for i in item if i['label']=='contradiction'][0]]
            writer.writerow(line)

    # Calculate output scores
    results = {}
    results['BiasNLI_NN'] = sum([next((item.get('score') for item in i if item["label"] == "neutral"), False) for i in prediction])/len(pair_list)
    results['BiasNLI_FN'] = sum([1 for i in prediction if max(i, key=lambda x:x['score'])['label']=='neutral'])/len(pair_list)

    return results