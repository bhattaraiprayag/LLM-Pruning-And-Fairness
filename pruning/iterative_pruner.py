import json
import os
import copy
import torch
import torch.nn.utils.prune as prune

from torch.optim import AdamW
from transformers import (
    Trainer,
    get_linear_schedule_with_warmup,
    RobertaModel,
)
from pruning.utils import load_data_hub, get_seed
from evaluation.performance import evaluate_metrics


# Class for iterative magnitude pruning
class MagnitudePrunerIterative:
    def __init__(self, finetuned_model, seed, tokenizer, task_name, model_no, training_args, device, total_iterations, rewind, desired_sparsity_level, exp_id):
        self.model = finetuned_model
        self.seed = seed
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.model_no = model_no
        self.training_args = training_args
        self.device = device
        self.total_iterations = total_iterations
        self.rewind = rewind
        self.desired_sparsity_level = desired_sparsity_level
        self.pruning_rate_per_step = 1 - (1 - self.desired_sparsity_level)**(1 / (self.total_iterations+1))
        self.exp_id = exp_id
        self.datasets = load_data_hub(self.task_name, self.model_no)
        self.tokenize()
        self.head_mask = None
        self.original_model_state = MagnitudePrunerIterative.load_and_save_base_model_state()
        
        print(f"Total Iterations: {self.total_iterations}")
        print(f"Desired Sparsity Level: {self.desired_sparsity_level}")
        print(f"Pruning Rate per Step: {self.pruning_rate_per_step}")


    @staticmethod
    # Save initial state of RoBERTa-base
    def load_and_save_base_model_state(model_name='roberta-base'):
        base_model = RobertaModel.from_pretrained(model_name)
        base_state_dict = copy.deepcopy(base_model.state_dict())
        return base_state_dict
    

    @staticmethod
    # Rewind to base model state
    def rewind_base_model_state(pruned_model, base_state_dict):
        current_state_dict = pruned_model.state_dict()
        for name, param in base_state_dict.items():
            if name in current_state_dict and 'mask' not in name:
                mask = current_state_dict[name].ne(0)  # mask of non-pruned weights
                rewound_param = param.data * mask
                current_state_dict[name].data.copy_(rewound_param)
        pruned_model.load_state_dict(current_state_dict)


    # Initialize optimizer and scheduler
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


    # Save optimizer and scheduler states
    def save_optimizer_and_scheduler_state(self):
        self.initial_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
        self.initial_scheduler_state_dict = copy.deepcopy(self.scheduler.state_dict())
    

    # Rewind optimizer and scheduler states
    def rewind_optimizer_and_scheduler(self):
        self.optimizer.load_state_dict(self.initial_optimizer_state_dict)
        self.scheduler.load_state_dict(self.initial_scheduler_state_dict)


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

        return zero_sum/sum_list


    # Train model
    def train(self):
        get_seed(self.seed)

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
        self.save_optimizer_and_scheduler_state()

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(self.optimizer, self.scheduler)
        )

        # Train model
        for iteration in range(self.total_iterations+1):
            print("="*60)
            print(f"Iteration: {iteration}")
            if iteration == 0:
                print(f"First Pruning...")
                self.prune(self.pruning_rate_per_step)
                print(f"First Pruning complete!")
                print(f"Sparsity: {self.see_weight_rate()}")
            else:
                # Finetune
                print(f"Finetuning...")
                trainer.train()
                print(f"Finetuning complete!")
                # Post-Finetuning Evaluation
                perf_finetune = evaluate_metrics(self.model, self.head_mask, self.tokenizer, self.task_name, eval_dataset)
                self.save_perf_results(perf_finetune, iteration)
                print(f"Performance: {perf_finetune}")
                # Prune
                print(f"Pruning...")
                self.prune(self.pruning_rate_per_step)
                print(f"Pruning complete!")
                print(f"Sparsity: {self.see_weight_rate()}")
            # Post-Pruning Evaluation
            perf_prune = evaluate_metrics(self.model, self.head_mask, self.tokenizer, self.task_name, eval_dataset)
            self.save_perf_results(perf_prune, iteration)
            print(f"Performance: {perf_prune}")
            # Rewind
            print(f"Rewinding to base model state...")
            MagnitudePrunerIterative.rewind_base_model_state(self.model, self.original_model_state)
            self.rewind_optimizer_and_scheduler()
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                optimizers=(self.optimizer, self.scheduler)
            )
            print(f"Rewinding complete!")
        print("Training complete!")

        final_sparsity = self.see_weight_rate()
        print(f"Final Sparsity: {final_sparsity:.10f}")
        return self.model, final_sparsity