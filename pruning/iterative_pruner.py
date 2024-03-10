import torch
import torch.nn.utils.prune as prune
import json
import os
import tqdm
import copy

from torch.optim import AdamW
from transformers import Trainer, get_linear_schedule_with_warmup
from pruning.utils import load_data_hub, get_seed
from evaluation.performance import evaluate_metrics


# Class for iterative magnitude pruning
class MagnitudePrunerIterative:
    def __init__(self, model, seed, tokenizer, task_name, model_no, training_args, device, total_iterations, rewind, desired_sparsity_level, exp_id):
        self.model = model
        self.seed = seed
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.model_no = model_no
        self.training_args = training_args
        self.device = device
        self.total_iterations = total_iterations
        self.rewind = rewind
        self.sparsity_level = desired_sparsity_level
        self.pruning_rate_per_step = desired_sparsity_level / (total_iterations + 1)
        self.exp_id = exp_id
        self.datasets = load_data_hub(self.task_name, self.model_no)
        self.tokenize()
        self.head_mask = None


    # Save initial state
    def save_initial_state(self):
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
        self.initial_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())


    # Rewind to initial state
    def rewind_weights(self):
        current_state_dict = self.model.state_dict()
        for name, param in self.initial_state_dict.items():
            if name in current_state_dict:
                mask = current_state_dict[name].ne(0)  # create a mask of which weights are currently not pruned
                # Only rewind the non-pruned weights
                rewound_param = param.data * mask
                current_state_dict[name].data.copy_(rewound_param)

        # Reload the optimizer state
        self.optimizer.load_state_dict(self.initial_optimizer_state_dict)


    # Initialize optimizer and scheduler within __init__ or another setup method
    def initialize_optimizer_and_scheduler(self, train_dataset):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            eps=self.training_args.adam_epsilon
        )
        total_steps = len(train_dataset) * self.training_args.num_train_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=total_steps
        )


    # Tokenize data
    def tokenize(self):
        def preprocess(examples):
            fields = ('premise', 'hypothesis') if self.task_name == 'mnli' else ('sentence1', 'sentence2')
            return self.tokenizer(*[examples[field] for field in fields], padding='max_length', max_length=128, truncation=True)
        self.datasets = {split: dataset.map(preprocess, batched=True) for split, dataset in self.datasets.items()}


    # Prune model
    def prune(self, pruning_rate):
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
        return self.model


    # Save evaluation results
    def save_perf_results(self, eval_results, iter):
        filename = f"{self.training_args.output_dir}/eval_iter_{iter}.json"
        if os.path.exists(filename):
            filename = f"{self.training_args.output_dir}/eval_iter_{iter}_post_pruning.json"
        with open(filename, 'w') as f:
            json.dump(eval_results, f)


    # See weight rate
    def see_weight_rate(self):
        sum_list = 0
        zero_sum = 0
        for ii in range(12):
            sum_list = sum_list+float(self.model.roberta.encoder.layer[ii].attention.self.query.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(self.model.roberta.encoder.layer[ii].attention.self.query.weight == 0))

            sum_list = sum_list+float(self.model.roberta.encoder.layer[ii].attention.self.key.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(self.model.roberta.encoder.layer[ii].attention.self.key.weight == 0))

            sum_list = sum_list+float(self.model.roberta.encoder.layer[ii].attention.self.value.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(self.model.roberta.encoder.layer[ii].attention.self.value.weight == 0))

            sum_list = sum_list+float(self.model.roberta.encoder.layer[ii].attention.output.dense.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(self.model.roberta.encoder.layer[ii].attention.output.dense.weight == 0))

            sum_list = sum_list+float(self.model.roberta.encoder.layer[ii].intermediate.dense.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(self.model.roberta.encoder.layer[ii].intermediate.dense.weight == 0))

            sum_list = sum_list+float(self.model.roberta.encoder.layer[ii].output.dense.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(self.model.roberta.encoder.layer[ii].output.dense.weight == 0))

        # If pooler is present
        if hasattr(self.model.roberta, 'pooler') and self.model.roberta.pooler is not None:
            sum_list = sum_list+float(self.model.roberta.pooler.dense.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(self.model.roberta.pooler.dense.weight == 0))

        return 100*zero_sum/sum_list


    # Train model
    def train(self):
        get_seed(self.seed)

        print(f"First Pruning (before finetuning)...")
        self.prune(self.pruning_rate_per_step)
        print(f"First Pruning complete!")
        print(f"Sparsity after first pruning: {self.see_weight_rate()}")

        # Debug statement (slice data if True)
        slicer = False
        if slicer:
            train_dataset = self.datasets["train"]
            train_dataset = train_dataset.select(range(5))
        else:
            train_dataset = self.datasets["train"]
        
        if self.task_name == 'mnli':
            eval_dataset = [self.datasets["validation_matched"], self.datasets["validation_mismatched"]]
        else:
            eval_dataset = self.datasets["validation"]

        # Initialize optimizer and scheduler
        self.initialize_optimizer_and_scheduler(train_dataset)

        # Save the initial state after the first pruning (but before fine-tuning)
        self.save_initial_state()

        for iteration in range(self.total_iterations):
            print("=====================================================")
            print(f'Iteration: {iteration+1}')

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                optimizers=(self.optimizer, self.scheduler)
            )

            # Finetune
            print(f'Finetuning...')
            trainer.train()
            print(f'Finetuning complete!')
            print("=====================")

            # Post-Finetuning Evaluation
            perf_finetune = evaluate_metrics(self.model, self.head_mask, self.tokenizer, self.task_name, eval_dataset)
            self.save_perf_results(perf_finetune, iteration+1)
            print(f"Performance after finetuning: {perf_finetune}")
            print(f"Sparsity after finetuning: {self.see_weight_rate()}")
            print("=====================")

            # Prune
            print(f"Pruning...")
            self.prune(self.pruning_rate_per_step)
            print(f"Pruning complete!")
            print("=====================")

            # Post-Pruning Evaluation
            perf_prune = evaluate_metrics(self.model, self.head_mask, self.tokenizer, self.task_name, eval_dataset)
            self.save_perf_results(perf_prune, iteration+1)
            print(f"Performance after pruning: {perf_prune}")
            print(f"Sparsity after pruning: {self.see_weight_rate()}")
            print("=====================")
            
            # Rewind weights if required
            if self.rewind:
                print("Rewinding to initial state...")
                self.rewind_weights()
                print("Rewinding complete!")

            # Reinitialize the trainer with reset optimizer and scheduler
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                optimizers=(self.optimizer, self.scheduler)
            )
        
        final_sparsity = self.see_weight_rate()
        print(f"Final Sparsity: {final_sparsity}%")
        
        return self.model, final_sparsity