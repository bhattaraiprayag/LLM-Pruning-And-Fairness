import os
from pruning.magnitude_pruner import MagnitudePrunerOneShot
from pruning.iterative_pruner import MagnitudePrunerIterative
from pruning.structured_pruning import structured_pruning

from pruning.sparsity_check import analyse_sparsity

from transformers import TrainingArguments


def pruning(exp_args, model, tokenizer, exp_id, experiment_dir):
    returned_sparsity = None
    head_mask = None

    if exp_args.pruning_method == 'structured':
        returned_sparsity, head_mask = structured_pruning(model, tokenizer, exp_args.seed, exp_args.task, exp_args.device,
                                               exp_args.masking_threshold, exp_id, exp_args.model_no)
    elif exp_args.pruning_method == 'imp':
        # Fine-tuning arguments
        training_args = TrainingArguments(
            output_dir=f'results/run{exp_id}',
            num_train_epochs=3,
            per_device_train_batch_size=32 if exp_args.task == 'mnli' else 8,
            per_device_eval_batch_size=32 if exp_args.task == 'mnli' else 8,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            adam_epsilon=1e-8
        )
        # Pruning arguments
        total_iterations=10
        rewind=True

        pruner = MagnitudePrunerIterative(model, exp_args.seed, tokenizer, exp_args.task, exp_args.model_no, training_args, exp_args.device, total_iterations, rewind, exp_args.sparsity_level, exp_id)
        pruner.train()
        returned_sparsity = exp_args.sparsity_level
    else:
        pruner = MagnitudePrunerOneShot(model, exp_args.seed, exp_args.pruning_method, exp_args.sparsity_level)
        pruner.prune()
        returned_sparsity = exp_args.sparsity_level

    # save pruned model
    pruned_model_dir = f'{experiment_dir}/pruned_model/'
    if not os.path.exists(pruned_model_dir):
        os.makedirs(pruned_model_dir)
    model.save_pretrained(pruned_model_dir)
    tokenizer.save_pretrained(pruned_model_dir)

    return returned_sparsity, head_mask
