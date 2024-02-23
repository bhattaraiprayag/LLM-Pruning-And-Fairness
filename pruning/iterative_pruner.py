import torch
import torch.nn.utils.prune as prune
import copy

from transformers import RobertaForSequenceClassification
from pruning.utils import get_seed, check_sparsity

# Class for iterative magnitude pruning
class MagnitudePrunerIterative:
    def __init__(self, model, seed, sparsity_level=0.3, total_iterations=10):
        self.model = model
        self.seed = seed
        self.sparsity_level = sparsity_level
        self.total_iterations = total_iterations
        self.seed = get_seed(seed)
        self.orig_model_state_dict = self._rewind_state_dict(model.state_dict())
    
    def _rewind_state_dict(self, state_dict):
        rewind_dict = {}
        for key in state_dict.keys():
            if 'roberta' in key:
                new_key = key + '_orig' if 'weight' in key else key
                rewind_dict[new_key] = state_dict[key]
        return rewind_dict

    def prune_model(self):
        for iteration in range(self.total_iterations):
            px = self._calculate_pruning_amount(iteration)
            self._perform_pruning(px)
            self._rewind_weights()
            sparsity = check_sparsity(self.model)
            print(f"Iteration {iteration + 1}/{self.total_iterations}, Sparsity: {sparsity}")

    def _calculate_pruning_amount(self, iteration):
        return self.sparsity_level * (iteration + 1) / self.total_iterations

    def _perform_pruning(self, px):
        parameters_to_prune = []
        for layer in range(12):
            layers = [
                self.model.roberta.encoder.layer[layer].attention.self.query,
                self.model.roberta.encoder.layer[layer].attention.self.key,
                self.model.roberta.encoder.layer[layer].attention.self.value,
                self.model.roberta.encoder.layer[layer].attention.output.dense,
                self.model.roberta.encoder.layer[layer].intermediate.dense,
                self.model.roberta.encoder.layer[layer].output.dense,
            ]
            parameters_to_prune += [(layer, 'weight') for layer in layers]

        parameters_to_prune.append((self.model.roberta.pooler.dense, 'weight'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=px)

    def _rewind_weights(self):
        model_dict = self.model.state_dict()
        model_dict.update(self.orig_model_state_dict)
        self.model.load_state_dict(model_dict)