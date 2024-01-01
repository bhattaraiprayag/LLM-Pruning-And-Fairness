import os
import json
import csv
import transformers

thisdir = os.path.dirname(os.path.realpath(__file__))
experiment_id = 'test_biasnli'

# Load model
model = transformers.RobertaForSequenceClassification.from_pretrained("models/MNLI/", use_safetensors=True, local_files_only=True)
tokenizer = transformers.AutoTokenizer.from_pretrained("models/MNLI/")
pipe = transformers.TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)

# Load csv with test sentences
with open('evaluation/data/bias_nli/occupation_gender.csv', mode='r') as csv_file:
    test_dict = csv.DictReader(csv_file)
    line_count = 0
    nn_count = 0
    fn_count = 0
    for row in test_dict:
        line_count += 1
        if line_count % 1000 == 0:
            print(line_count)
        prediction = pipe((row["premise"],row["hypothesis"]))
        ent = [i['score'] for i in prediction if i['label']=='entailment'][0]
        neut = [i['score'] for i in prediction if i['label']=='neutral'][0]
        con = [i['score'] for i in prediction if i['label']=='contradiction'][0]
        nn_count += neut
        if neut == max(ent, neut, con):
            fn_count += 1
    results = {}
    results['fraction_neutral'] = fn_count/line_count
    results['net_neutral'] = nn_count / line_count
    with open(f"{thisdir}/evaluation/results/seat/{experiment_id}.json", "w") as f:
        json.dump(results, f)

