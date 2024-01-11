import os
import re
import numpy as np
import torch
import torch.nn as nn
import logging
from transformers import RobertaModel, RobertaConfig, RobertaForMaskedLM
import math
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer

logger = logging.getLogger(__name__)
ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin"

}

ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "swish": lambda x: x * F.sigmoid(x),
    # Add other activation functions as needed
}

RobertaLayerNorm = torch.nn.LayerNorm


def load_tf_weights_in_roberta(model, config, tf_checkpoint_path):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Loading a TensorFlow model in PyTorch requires TensorFlow to be installed. Please see "
                          "https://www.tensorflow.org/install/ for installation instructions.")

    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))

    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")

        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            print("Skipping {}".format("/".join(name)))
            continue

        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue

            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise

        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    return model

class RobertaEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=1)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# Example usage:
config = RobertaConfig.from_pretrained('roberta-base')
roberta_model = RobertaEmbeddings(config)
tf_checkpoint_path = '/Applications/Marioma/UNI/Team Project/final_models/STS-B'
roberta_model = load_tf_weights_in_roberta(roberta_model, config, tf_checkpoint_path)

class RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs, value_layer) if self.output_attentions else (context_layer,)
        return outputs


class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        if self.dense:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        if self.self is None:
            return

        all_pruned = self.pruned_heads.union(heads)
        if len(all_pruned) == self.self.num_attention_heads:
            self.self = None
            self.output.dense = None
            # Update hyper params and store pruned heads
            self.pruned_heads = all_pruned
            return

        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if self.self is None:
            return (self.output(None, hidden_states),)
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        if len(self_outputs) == 1:
            outputs = (attention_output,)
        else:
            _, attention_probs, value_layer = self_outputs
            output_head_weights = self.output.dense.weight.view(self.self.num_attention_heads, self.self.attention_head_size, -1)
            proj_up_value_layer = torch.einsum("ijkl,jld->ijkd", value_layer, output_head_weights)
            alpha_f =  torch.einsum("ijkl,ijld->ijkld", attention_probs, proj_up_value_layer)
            outputs = (attention_output, attention_probs, alpha_f)
        return outputs

class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = torch.nn.functional.gelu  # Assuming GELU activation for RoBERTa

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, mlp_mask):
        if self.dense is None:
            hidden_states = self.LayerNorm(input_tensor)
        else:
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            if mlp_mask is not None:
                hidden_states = hidden_states * mlp_mask
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def prune_mlp(self):
        self.intermediate = None
        self.output.dense = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        mlp_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights
        if self.intermediate is None:
            intermediate_output = None
        else:
            intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, mlp_mask)
        outputs = (layer_output,) + outputs
        return outputs


class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            mlp_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        all_alpha_f_s = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], mlp_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                if len(layer_outputs) > 1:
                    all_attentions = all_attentions + (layer_outputs[1],)

                    all_alpha_f_s = all_alpha_f_s + (layer_outputs[2],)
                else:
                    all_attentions = all_attentions + (None,)
                    all_alpha_f_s = all_alpha_f_s + (None,)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions, all_alpha_f_s, attention_mask)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RobertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = RobertaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RobertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RobertaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class RobertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RobertaLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RobertaOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class RobertaPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RobertaLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class RobertaPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = RobertaConfig  # Update with the correct config class for RoBERTa
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP  # Update with the correct map
    load_tf_weights = load_tf_weights_in_roberta  # Implement a function to load RoBERTa TF weights
    base_model_prefix = "roberta"  # Update with the correct prefix for RoBERTa

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RobertaLayerNorm):  # Update with the correct LayerNorm class for RoBERTa
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class RobertaModelCustom(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Customize or add any additional layers you need for your specific task
        self.pooler = RobertaPooler(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def prune_mlps(self, mlps_to_prune):
        self.config.pruned_mlps = mlps_to_prune
        for layer in mlps_to_prune:
            self.encoder.layer[layer].prune_mlp()

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RobertaModelCustom(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Customize or add any additional layers you need for your specific task
        self.pooler = RobertaPooler(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            mlp_mask=None,
    ):
        """
        Return:
            Tuple of torch.FloatTensor comprising various elements depending on the configuration (RoBertaConfig) and inputs:
            last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function.
                The Linear layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.
            hidden_states (tuple(torch.FloatTensor), optional, returned when config.output_hidden_states=True):
                Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer)
                of shape (batch_size, sequence_length, hidden_size).
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (tuple(torch.FloatTensor), optional, returned when config.output_attentions=True):
                Tuple of torch.FloatTensor (one for each layer) of shape
                (batch_size, num_heads, sequence_length, sequence_length).
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        # Input validation
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Attention mask handling
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # Encoder attention mask handling
        if encoder_hidden_states is not None and encoder_attention_mask is None:
            encoder_attention_mask = torch.ones_like(encoder_hidden_states)

        # MLP mask handling
        if mlp_mask is None:
            mlp_mask = [None] * self.config.num_hidden_layers
        else:
            mlp_mask = mlp_mask.to(dtype=next(self.parameters()).dtype)

        # ... (other setup code)

        # Embedding Layer
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )

        # Encoder Layer
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            mlp_mask=mlp_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # Pooling Layer
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # Construct Output
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

class RobertaForMaskedLM(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.cls = RobertaOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        """
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
            loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Total loss as the masked language modeling loss.
            prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (loss), prediction_scores, (hidden_states), (attentions)

    class RobertaForNextSentencePrediction(RobertaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)

            self.roberta = RobertaModel(config)
            self.cls = RobertaOnlyNSPHead(config)

            self.init_weights()


        def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                next_sentence_label=None,
        ):
            """
            next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
                Indices should be in ``[0, 1]``.
                ``0`` indicates sequence B is a continuation of sequence A,
                ``1`` indicates sequence B is a random sequence.

            Returns:
                :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
                loss (`optional`, returned when ``next_sentence_label`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                    Total loss as the next sequence prediction (classification) loss.
                seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, 2)`):
                    Prediction scores of the next sequence prediction (classification) head (scores of True/False
                    continuation before SoftMax).
                hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
                    Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                    of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
                attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                    Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                    :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                    heads.
            """
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            pooled_output = outputs[1]
            seq_relationship_score = self.cls(pooled_output)

            outputs = (seq_relationship_score,) + outputs[2:]  # Add hidden states and attention if they are here

            if next_sentence_label is not None:
                loss_fct = CrossEntropyLoss()
                next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                outputs = (next_sentence_loss,) + outputs

            return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)

class RobertaForSequenceClassification(RobertaPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.num_labels = config.num_labels

                self.roberta = RobertaModel(config)
                self.dropout = nn.Dropout(config.hidden_dropout_prob)
                self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

                self.init_weights()

            def forward(
                    self,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    mlp_mask=None,
                    inputs_embeds=None,
                    labels=None,
            ):
                r"""
                labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                    Labels for computing the sequence classification/regression loss.
                    Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                    If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                    If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

                Returns:
                    :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
                    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                        Classification (or regression if config.num_labels==1) loss.
                    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                        Classification (or regression if config.num_labels==1) scores (before SoftMax).
                    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                        Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                        of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
                    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                        Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                        :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                        heads.
                """
                outputs = self.roberta(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    mlp_mask=mlp_mask,
                    inputs_embeds=inputs_embeds,
                )

                pooled_output = outputs[1]

                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)

                outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

                if labels is not None:
                    if self.num_labels == 1:
                        #  We are doing regression
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), labels.view(-1))
                    else:
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    outputs = (loss,) + outputs

                return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForMultipleChoice(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
            loss (:obj:`torch.FloatTensor`` of shape ``(1,)`, `optional`, returned when :obj:`labels` is provided):
                Classification loss.
            classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
                `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

                Classification scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


    class RobertaForTokenClassification(RobertaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels

            self.roberta = RobertaModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

            self.init_weights()

        def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
        ):
            r"""
            Same as the forward method in BertForTokenClassification.
            """
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            sequence_output = outputs[0]

            sequence_output = self.dropout(sequence_output)
            logits = self.classifier(sequence_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs  # (loss), scores, (hidden_states), (attentions)


class RobertaForQuestionAnswering(RobertaPreTrainedModel):
        def __init__(self, config):
            super(RobertaForQuestionAnswering, self).__init__(config)
            self.num_labels = config.num_labels

            self.roberta = RobertaModel(config)
            self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

            self.init_weights()

       # @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
        def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None,
        ):
            r"""
            Same as the forward method in BertForQuestionAnswering.
            """
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            sequence_output = outputs[0]

            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (start_logits, end_logits,) + outputs[2:]
            if start_positions is not None and end_positions is not None:
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

            return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
