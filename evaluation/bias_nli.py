import os
import json
import csv
import transformers

thisdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_id = 'test_biasnli'

# Load model
model = transformers.RobertaForSequenceClassification.from_pretrained(f"{thisdir}/models/MNLI/", use_safetensors=True, local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(f"{thisdir}/models/MNLI/")
pipe = transformers.TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, max_length=512, truncation=True, padding=True)

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
    os.makedirs(f"{thisdir}/evaluation/results/bias_nli", exist_ok=True)
    with open(f"{thisdir}/evaluation/results/bias_nli/{experiment_id}.json", "w") as f:
        json.dump(results, f)

