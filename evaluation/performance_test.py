import numpy as np
import torch
from performance import load_test_dataset, evaluate_metrics

from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification
)

# load model
model_path = '/Users/mariamamir/TeamProject/final_models/mnli'

model = RobertaForSequenceClassification.from_pretrained(
    model_path,
    use_safetensors=True,
    local_files_only=True
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load head mask
file_path = '/Users/mariamamir/TeamProject/LLM-Pruning-And-Fairness/head_mask.npy'
head_mask = np.load(file_path, allow_pickle=True)

# Replace Ellipsis with 0 (or any other appropriate value)
head_mask[head_mask == Ellipsis] = 0

# Check and convert NumPy array to numeric data
if head_mask.dtype == object:
    try:
        head_mask = head_mask.astype(np.float32)
    except ValueError as e:
        print(f"Error converting head_mask to numeric data: {e}")
        exit()

# Convert NumPy array to PyTorch tensor
head_mask_tensor = torch.tensor(head_mask, dtype=torch.float32)

# evaluate model "performance" (not fairness)
eval_datasets = load_test_dataset('mnli', 2)
res_performance = evaluate_metrics(model, head_mask_tensor, tokenizer, 'mnli', eval_datasets)
print(res_performance)