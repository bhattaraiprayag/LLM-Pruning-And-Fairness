import torch
import torch.nn.utils.prune as prune
import random
import numpy as np

from pruning.utils import get_seed, check_sparsity
from pruning.checker import analyse_sparsity

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

        # Check sparsity before pruning
        print(f"Sparsity before pruning: {analyse_sparsity(self.model)}")

        if self.pruning_method == "l1-unstructured":
            self._apply_l1_unstructured()
        elif self.pruning_method == "l1-unstructured-linear":
            self._apply_l1_unstructured_linear()
        elif self.pruning_method == "random-unstructured":
            self._apply_random_unstructured()
        elif self.pruning_method == "global-unstructured":
            self._apply_global_unstructured()
        elif self.pruning_method == "global-unstructured-attention":
            self._apply_global_unstructured_attention()
        else:
            raise ValueError("Magnitude Pruning method not supported, yet!")

        # Check sparsity after pruning
        print(f"Sparsity after pruning: {analyse_sparsity(self.model)}")

        return self.model


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
    

    # Method 3: Global unstructured pruning
    def _apply_global_unstructured(self):
        # Collect parameters to prune
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                parameters_to_prune.append((module, 'weight'))

        # Convert to tuple as required by prune.global_unstructured
        parameters_to_prune = tuple(parameters_to_prune)

        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.sparsity_level
        )

        # Remove the reparametrization after pruning
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')


    # Method 4: Global unstructured pruning for attention heads
    def _apply_global_unstructured_attention(self):
        parameters_to_prune = []
        
        if hasattr(self.model, 'roberta'):
            # Prune specific layers within each encoder layer
            for layer in self.model.roberta.encoder.layer:
                layers = [
                    layer.attention.self.query,
                    layer.attention.self.key,
                    layer.attention.self.value,
                    layer.attention.output.dense,
                    layer.intermediate.dense,
                    layer.output.dense,
                ]
                parameters_to_prune.extend([(l, 'weight') for l in layers])

            # Prune the pooler layer, if it exists
            if hasattr(self.model.roberta, 'pooler') and self.model.roberta.pooler is not None:
                parameters_to_prune.append((self.model.roberta.pooler.dense, 'weight'))

        # Convert to tuple as required by prune.global_unstructured
        parameters_to_prune = tuple(parameters_to_prune)

        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.sparsity_level
        )

        # Remove the reparametrization after pruning
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')