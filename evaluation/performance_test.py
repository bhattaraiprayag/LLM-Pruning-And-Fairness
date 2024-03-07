import numpy as np
import torch
from evaluation.performance import load_test_dataset, evaluate_metrics
from pruning.structured_pruning import structured_pruning
from pruning.utils import get_device

from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification
)

device = get_device()
task = 'mnli'

# load model
model_path = 'training/final_models/MNLI/model_no1/'

model = RobertaForSequenceClassification.from_pretrained(
    model_path,
    use_safetensors=True,
    local_files_only=True
)

model.to(device)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# get head_mask
sparsity, head_mask = structured_pruning(model, tokenizer, 1, task, 0, 0.95, 999, 1)

# evaluate model "performance" (not fairness)
eval_datasets = load_test_dataset(task, 1)
res_performance = evaluate_metrics(model, head_mask, tokenizer, task, eval_datasets)
print(res_performance)