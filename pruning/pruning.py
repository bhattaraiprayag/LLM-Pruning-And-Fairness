import os
from pruning.magnitude_pruner import MagnitudePrunerOneShot
from pruning.structured_pruning import structured_pruning


def pruning(exp_args, model, tokenizer, exp_id, experiment_dir):
    returned_sparsity = None

    if exp_args.pruning_method == 'structured':
        returned_sparsity = structured_pruning(model, tokenizer, exp_args.seed, exp_args.task, exp_args.device,
                                               exp_args.masking_threshold, exp_id, exp_args.model_no)
    elif exp_args.pruning_method == 'imp':
        # add IMP
        # returned_sparsity =
        raise NotImplementedError("Not implemented, yet!")
    else:
        pruner = MagnitudePrunerOneShot(model, exp_args.seed, exp_args.pruning_method, exp_args.sparsity_level)
        pruner.prune()

    # save pruned model
    pruned_model_dir = f'{experiment_dir}/pruned_model/'
    if not os.path.exists(pruned_model_dir):
        os.makedirs(pruned_model_dir)
    model.save_pretrained(pruned_model_dir)
    tokenizer.save_pretrained(pruned_model_dir)

    return returned_sparsity
