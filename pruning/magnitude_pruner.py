import torch
import torch.nn.utils.prune as prune
import random
import numpy as np

from utils import set_seed, check_sparsity

class MagnitudePrunerOneShot:
    def __init__(self, model, seed, pruning_method, sparsity_level=0.3):
        self.model = model
        self.seed = seed
        self.pruning_method = pruning_method
        self.sparsity_level = sparsity_level

    def prune(self):
        if self.pruning_method == "l1-unstructured":
            self._apply_l1_unstructured()
        elif self.pruning_method == "l1-unstructured-linear":
            self._apply_l1_unstructured_linear()
        elif self.pruning_method == "l1-unstructured-invert":
            self._apply_l1_unstructured_invert()
        else:
            raise ValueError("Pruning method not supported, yet!")
    
    # Method 1: L1-unstructured (Remove weights globally)
    def _apply_l1_unstructured(self):
        # Set seed
        set_seed(self.seed)

        # Check sparsity before pruning
        print(f"Sparsity before pruning: {check_sparsity(self.model):.4%}")

        # Pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                prune.l1_unstructured(module, name='weight', amount=self.sparsity_level)
        
        # Remove the reparametrization after pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                prune.remove(module, 'weight')
        
        # Check sparsity after pruning
        print(f"Sparsity after pruning: {check_sparsity(self.model):.4%}")


    # Method 2: L1-unstructured (Remove weights of linear layers only)
    def _apply_l1_unstructured_linear(self):
        # Set seed
        set_seed(self.seed)

        # Check sparsity before pruning
        print(f"Sparsity before pruning: {check_sparsity(self.model):.4%}")

        # Pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.sparsity_level)
        
        # Remove the reparametrization after pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')
        
        # Check sparsity after pruning
        print(f"Sparsity after pruning: {check_sparsity(self.model):.4%}")


    # Method 3: L1-unstructured-invert (Not implemented, yet!)
    def _apply_l1_unstructured_invert(self):
        # TODO: implement
        raise NotImplementedError("Not implemented, yet!")