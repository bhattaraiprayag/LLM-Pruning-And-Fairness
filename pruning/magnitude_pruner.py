import torch
import torch.nn.utils.prune as prune
import random
import numpy as np

from pruning.utils import get_seed, check_sparsity

# Class for one-shot magnitude pruning
class MagnitudePrunerOneShot:
    def __init__(self, model, seed, pruning_method, sparsity_level=0.3):
        self.model = model
        self.seed = seed
        self.pruning_method = pruning_method
        self.sparsity_level = sparsity_level

    def prune(self):
        # Set seed
        get_seed(self.seed)

        # # Check sparsity before pruning
        # print(f"Sparsity before pruning: {check_sparsity(self.model):.4%}")

        if self.pruning_method == "l1-unstructured":
            self._apply_l1_unstructured()
        elif self.pruning_method == "l1-unstructured-linear":
            self._apply_l1_unstructured_linear()
        elif self.pruning_method == "l1-unstructured-invert":
            self._apply_l1_unstructured_invert()
        elif self.pruning_method == "random-unstructured":
            self._apply_random_unstructured()
        else:
            raise ValueError("Pruning method not supported, yet!")
        
        # Check sparsity after pruning
        print(f"Sparsity after pruning: {check_sparsity(self.model):.4%}")

        return self.model


    # Method 1: L1-unstructured (Remove weights globally)
    def _apply_l1_unstructured(self):
        # Pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                prune.l1_unstructured(module, name='weight', amount=self.sparsity_level)
        
        # Remove the reparametrization after pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                prune.remove(module, 'weight')


    # Method 2: L1-unstructured (Remove weights of linear layers only)
    def _apply_l1_unstructured_linear(self):
        # Pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.sparsity_level)

        # Remove the reparametrization after pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')


    # Method 3: L1-unstructured-invert (Not implemented, yet!)
    def _apply_l1_unstructured_invert(self):
        # TODO: implement
        raise NotImplementedError("Not implemented, yet!")

    # Baseline method: Random pruning
    def _apply_random_unstructured(self):
        # Pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                prune.random_unstructured(module, name='weight', amount=self.sparsity_level)

        # Remove the reparametrization after pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                prune.remove(module, 'weight')