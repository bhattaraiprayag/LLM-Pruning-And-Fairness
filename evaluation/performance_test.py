import numpy as np

from performance import load_test_dataset, evaluate_metrics

from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification
)

# load model
model_path = '../training/final_models/MNLI/model_no2/'

model = RobertaForSequenceClassification.from_pretrained(
    model_path,
    use_safetensors=True,
    local_files_only=True
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load head mask
file_path = '../results/run1/s-pruning/head_mask.npy'
head_mask = np.load(file_path, allow_pickle=True)

# evaluate model "performance" (not fairness)
eval_datasets = load_test_dataset('mnli', 2)
res_performance = evaluate_metrics(model, head_mask, tokenizer, 'mnli', eval_datasets)
print(res_performance)