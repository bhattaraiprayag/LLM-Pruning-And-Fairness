import torch
import numpy as np

#file_path = '/Users/mariamamir/TeamProject/LLM-Pruning-And-Fairness/head_mask.npy'
#head_mask = np.load(file_path, allow_pickle=True)


def calculate_sparsity(data):
  """
  Calculates the sparsity of a head mask.

  Args:
      data: A NumPy array representing the head mask data.

  Returns:
      A float representing the sparsity level of the head mask (0 to 1).
  """
  total_elements = data.size
  non_zero_elements = np.count_nonzero(data != 0)  # Count non-zero elements efficiently
  sparsity = 1 - (non_zero_elements / total_elements)
  return sparsity


# Load the data assuming it's a text file with space-separated values
file_path = '/Users/mariamamir/TeamProject/LLM-Pruning-And-Fairness/head_mask.npy'
data = np.load(file_path, allow_pickle=True)

# Ensure the loaded data is a NumPy array (optional)
if not isinstance(data, np.ndarray):
  raise ValueError("Data must be a NumPy array.")

# Calculate sparsity
sparsity = calculate_sparsity(data)

# Print the overall sparsity
print(f"Overall sparsity level: {sparsity:.4f}")