
import numpy as np

file_path = '/Users/mariamamir/TeamProject/LLM-Pruning-And-Fairness/head_mask.npy'
head_masks = np.load(file_path, allow_pickle=True)

#Overall sparsity level: This refers to the percentage of the model's parameters or connections that are inactive.
def calculate_sparsity(head_masks):

  if head_masks.ndim == 1:  # Check if it's a 1D array (single mask)
    total_elements = len(head_masks)
    masked_elements = np.count_nonzero(head_masks == 0)
    sparsity = masked_elements / total_elements
  else:  # Handle 2D array (multiple masks)
    total_heads = head_masks.shape[1]
    total_masked_heads = np.sum(head_masks == 0, axis=1)
    total_masked_elements = np.sum(total_masked_heads)
    elements_per_head = head_masks.shape[0]
    total_elements = total_heads * elements_per_head
    sparsity = total_masked_elements / total_elements

  return sparsity
# 0 = No sparsity. [All elements in the model are used, and none are set to zero. This is the least sparse scenario]
# 1 = Complete sparsity. [All elements in the model are set to zero, and the model is essentially inactive]
# Values between 0 and 1 [Represent different levels of sparsity. Higher values (closer to 1) indicate that a larger portion of the model's elements are inactive and not contributing to the computations]

# Ensure head_masks is a NumPy array
if not isinstance(head_masks, np.ndarray):
  raise ValueError("head_masks must be a NumPy array.")

sparsity = calculate_sparsity(head_masks)
print(f"Overall sparsity level: {sparsity:.4f}")