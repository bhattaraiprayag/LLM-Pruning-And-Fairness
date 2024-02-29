import torch
import torch.nn.utils.prune as prune
import json
import os

from transformers import Trainer
from pruning.utils import load_data_hub, check_sparsity
from evaluation.performance import evaluate_metrics

# Class for iterative magnitude pruning
class MagnitudePrunerIterative:
    def __init__(self, model, tokenizer, task_name, model_no, training_args, device, total_iterations, rewind, pruning_rate_per_step, sparsity_level, exp_id):
        self.model = model
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.model_no = model_no
        self.training_args = training_args
        self.device = device
        self.total_iterations = total_iterations
        self.rewind = rewind
        self.pruning_rate_per_step = pruning_rate_per_step
        self.sparsity_level = sparsity_level
        self.exp_id = exp_id
        self.datasets = load_data_hub(self.task_name, self.model_no)
        self.initial_checkpoint = model.state_dict()  # Assuming initial model state is stored
        self.tokenize_data()

    def tokenize_data(self):
        def preprocess_data(examples):
            args = (
                (examples['sentence1'],) if 'sentence2' not in examples else (examples['sentence1'], examples['sentence2'])
            )
            return self.tokenizer(*args, padding='max_length', max_length=128, truncation=True)
        self.datasets = {split: dataset.map(preprocess_data, batched=True) for split, dataset in self.datasets.items()}

    def prune_model(self, pruning_rate):
        parameters_to_prune = []
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

        if hasattr(self.model.roberta, 'pooler') and self.model.roberta.pooler is not None:
            parameters_to_prune.append((self.model.roberta.pooler.dense, 'weight'))

        parameters_to_prune = tuple(parameters_to_prune)

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_rate,
        )

        # Debugging: Check a few weights before and after pruning
        for layer in self.model.roberta.encoder.layer:
            print(f"Layer {layer}: Weight sample before pruning: {layer.attention.self.query.weight.data[:5]}")
            print("Sparsity in layer: {:.2f}%".format(
                100. * float(torch.sum(layer.attention.self.query.weight == 0))
                / float(layer.attention.self.query.weight.nelement())
            ))

    def rewind_weights(self, checkpoint):
        pruned_model = self.model

        # Debugging: Store a copy of weights before rewinding
        pre_rewind_weights = {name: param.clone() for name, param in pruned_model.named_parameters()}

        for name in checkpoint.keys():
            if 'weight_orig' not in name:
                pruned_model.state_dict()[name] = checkpoint[name]
        
        # Debugging: Compare pre- and post-rewind weights for a few parameters
        for name, pre_param in pre_rewind_weights.items():
            post_param = pruned_model.state_dict()[name]
            if torch.equal(pre_param, post_param):
                print(f"No change in weights for {name}")
            else:
                print(f"Weights changed for {name}")

        self.model = pruned_model

    def train_model(self):
        # Debug statement
        train_dataset = self.datasets["train"]
        slicer = 10
        train_dataset = train_dataset.select(range(slicer))

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            # train_dataset=self.datasets["train"],
            train_dataset=train_dataset,
            eval_dataset=self.datasets["validation"]
        )

        trainer.train()

    def evaluate_model(self):
        return evaluate_metrics(self.model, self.tokenizer, self.task_name, self.datasets["validation"], self.exp_id)
    
    def save_eval(self, model_state, evaluation_results, step):
        with open(f"{self.training_args.output_dir}/eval_iter_{step}.json", 'w') as f:  # Save evaluation results for each iteration
            json.dump(evaluation_results, f)

    def prune(self):
        eval_results = self.evaluate_model()
        self.save_eval(self.model.state_dict(), eval_results, step=0)

        for iteration in range(1, self.total_iterations + 1):
            print(f'Sparsity before pruning iter #{iteration}: {check_sparsity(self.model):.4%}')
            self.prune_model(self.pruning_rate_per_step)
            print(f'Sparsity after pruning iter #{iteration}: {check_sparsity(self.model):.4%}')

            if self.rewind:
                self.rewind_weights(self.initial_checkpoint)
                print(f"Rewinding to initial weights after pruning iteration #{iteration}")

            self.train_model()
            eval_results = self.evaluate_model()
            self.save_eval(self.model.state_dict(), eval_results, step=iteration)
        
        return self.model