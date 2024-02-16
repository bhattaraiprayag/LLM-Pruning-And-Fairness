# Function for the bias_nli evaluation
# To note: templates must be available at the correct filepath

import csv
import os
from tqdm import tqdm
from evaluation.utils.bias_sts import get_device

def bias_nli(model, tokenizer, exp_id):
    # Input: model and tokenizer with the transformers pipeline containing the model and tokenizer
    # Output: results dictionary with net neutral and fraction neutral scores, also saves csv with predictions

    device = get_device()

    # Load csv with test sentences, then make model predictions for each
    with open(f'evaluation/data/bias_nli/occupation_gender.csv', mode='r') as csv_file:
        test_dict = csv.DictReader(csv_file)
        pair_list = []
        word_list = []
        prediction = []
        # Get all premise/hypothesis pairs in a list
        for row in tqdm(test_dict, desc="Loading bias-nli test sentences"):
            pair_list.append((row["premise"],row["hypothesis"]))
            word_list.append([row['premise_filler_word'], row['hypothesis_filler_word'], row["premise"],row["hypothesis"]])

            if len(pair_list)==2000:


            # # # DEBUGGING SWITCH: Work with less rows
            # if len(pair_list) >= 1000:
            #     break

                # Make predictions with model

                inputs = tokenizer(pair_list, max_length=512, truncation=True, padding=True, return_tensors='pt')
                inputs.to(device)
                preds = model(**inputs).logits.softmax(dim=1)
                prediction.extend(preds.tolist())
                pair_list = []

        inputs = tokenizer(pair_list, max_length=512, truncation=True, padding=True, return_tensors='pt')
        inputs.to(device)
        preds = model(**inputs).logits.softmax(dim=1)
        prediction.extend(preds.tolist())

    # Save predictions
    os.makedirs(f'results/run{exp_id}', exist_ok=True)
    with open(f'results/run{exp_id}/bias_nli.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["premise_filler_word", "hypothesis_filler_word", "premise", "hypothesis", "contradiction", "neutral", "entailment"])
        for i, item in enumerate(prediction):
            line = word_list[i] + item
            writer.writerow(line)

    # Calculate output scores
    results = {}
    results['BiasNLI_NN'] = [sum(i) for i in zip(*prediction)][1]/len(word_list)
    results['BiasNLI_FN'] = sum([1 for i in prediction if max(i)==i[1]])/len(word_list)

    return results