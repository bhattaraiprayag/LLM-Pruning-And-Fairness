def identify_layer_sparsity(layer):
    if hasattr(layer, 'weight'):
        weight = layer.weight.data
        total_elements = weight.numel()
        zero_elements = (weight == 0).sum()
        return zero_elements.item(), total_elements
    else:
        return 0, 0

def identify_pruned_heads(head_mask, num_attention_heads):
    pruned_heads = []
    if head_mask is not None:
        for head_index in range(num_attention_heads):
            if head_mask[head_index] == 0:
                pruned_heads.append(head_index)
    return pruned_heads

def analyse_sparsity(model, head_mask=None, verbose=False):
    total_zero_elements = 0
    total_elements = 0

    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    head_size = hidden_size // num_attention_heads

    # Handle both 'RobertaModel' and 'RobertaForSequenceClassification' models
    encoder = model.roberta.encoder if hasattr(model, 'roberta') else model.encoder if hasattr(model, 'encoder') else None
    if encoder is None:
        raise ValueError("The model does not have an encoder attribute.")

    for i, layer in enumerate(encoder.layer):
        current_head_mask = head_mask[i] if head_mask is not None and i < len(head_mask) else None
        if verbose or current_head_mask is not None:
            print("-"*60)
            print(f"RobertaLayer #{i+1}")

        layer_total_elements = 0
        layer_zero_elements = 0

        pruned_heads = identify_pruned_heads(current_head_mask, num_attention_heads)
        if current_head_mask is not None:
            pruned_head_elements = len(pruned_heads) * head_size * hidden_size
            layer_zero_elements += pruned_head_elements
            layer_total_elements += num_attention_heads * head_size * hidden_size

        # Process each sub-layer of the RobertaLayer
        if hasattr(layer, 'attention'):
            for sub_layer_name in ['query', 'key', 'value', 'output.dense']:
                sub_layer = getattr(layer.attention.self, sub_layer_name, None) if 'output.dense' not in sub_layer_name else getattr(layer.attention.output, 'dense', None)
                if sub_layer is not None:
                    zeros, elements = identify_layer_sparsity(sub_layer)
                    if verbose:
                        print(f"Sub-layer: {sub_layer_name} | Sparsity: {100 * zeros / elements:.10f}")
                    layer_zero_elements += zeros
                    layer_total_elements += elements

        if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
            zeros, elements = identify_layer_sparsity(layer.intermediate.dense)
            if verbose:
                print(f"Intermediate.dense | Sparsity: {100 * zeros / elements:.10f}")
            layer_zero_elements += zeros
            layer_total_elements += elements

        if hasattr(layer, 'output') and hasattr(layer.output, 'dense'):
            zeros, elements = identify_layer_sparsity(layer.output.dense)
            if verbose:
                print(f"Output.dense | Sparsity: {100 * zeros / elements:.10f}")
            layer_zero_elements += zeros
            layer_total_elements += elements
        
        if current_head_mask is not None:
            print(f"# Attention Heads: {num_attention_heads}")
            print(f"# Pruned Attention Heads: {len(pruned_heads)}")
            if pruned_heads:
                pruned_heads_str = ', '.join(map(str, pruned_heads))
                print(f"Pruned Head IDs: {pruned_heads_str}")

        total_zero_elements += layer_zero_elements
        total_elements += layer_total_elements
        layer_sparsity = 100 * layer_zero_elements / layer_total_elements if layer_total_elements > 0 else 0
        if head_mask is None:
            if verbose:
                print(f"Layer Sparsity: {layer_sparsity:.10f}")

    overall_sparsity_percentage = 100 * total_zero_elements / total_elements if total_elements > 0 else 0
    
    if verbose:
        print("-"*60)
        print(f"Overall Model Sparsity: {overall_sparsity_percentage:.10f}")
        print("-"*60)

    return overall_sparsity_percentage.__round__(10)


# # USAGE:
# # PARAMETERS: model, head_mask=None, verbose=False
# # Model Options: base_model, unpruned_model, magnitude_pruned_model, (structure_pruned_model, structure_pruned_head_mask). Note: ONLY Structure Pruning requires both model and head_mask.
# # Example 1: analyse_sparsity(base_model, verbose=True)   # For base_model, unpruned_model, magnitude_pruned_model
# # Example 2: analyse_sparsity(structure_pruned_model, structure_pruned_head_mask, verbose=False)  # Only for structure_pruned_model


# # START LOADING THE MODELS
# import numpy as np
# from transformers import (
#     RobertaForSequenceClassification,
#     RobertaModel
# )

# # Load the un-pruned and pruned model, and its mask
# unpruned_model_path = '/home/bhattaraiprayag/projects/team_project_uma/05.03.2024/training/final_models/STS-B/model_no2'
# structure_pruned_model_path = '/home/bhattaraiprayag/projects/team_project_uma/05.03.2024/results/run147/pruned_model/'
# structure_pruned_mask_path = '/home/bhattaraiprayag/projects/team_project_uma/05.03.2024/results/run147/s-pruning/head_mask_9.npy'
# magnitude_pruned_model_path = '/home/bhattaraiprayag/projects/team_project_uma/05.03.2024/results/run148/pruned_model/'

# base_model = RobertaModel.from_pretrained('roberta-base')
# unpruned_model = RobertaForSequenceClassification.from_pretrained(
#     unpruned_model_path,
#     use_safetensors=True,
#     local_files_only=True
# )
# structure_pruned_model = RobertaForSequenceClassification.from_pretrained(
#     structure_pruned_model_path,
#     use_safetensors=True,
#     local_files_only=True
# )
# structure_pruned_head_mask = np.loadtxt(structure_pruned_mask_path)
# magnitude_pruned_model = RobertaForSequenceClassification.from_pretrained(
#     magnitude_pruned_model_path,
#     use_safetensors=True,
#     local_files_only=True
# )

# analyse_sparsity(base_model, verbose=True)