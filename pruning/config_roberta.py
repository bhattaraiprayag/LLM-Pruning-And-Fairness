from transformers.configuration_utils import PretrainedConfig

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # Map of pre-trained RoBERTa model names to their configuration file URLs
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json",
    # Add more entries for other RoBERTa model variations if needed
}

class RobertaConfig(PretrainedConfig):
    r"""
    Configuration class for RoBERTa.

    Args:
        vocab_size (:obj:`int`, optional, defaults to 50265):
            Vocabulary size of the RoBERTa model. Defines the different tokens that
            can be represented by the `input_ids` passed to the forward method of :class:`~transformers.RobertaModel`.
        hidden_size (:obj:`int`, optional, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, optional, defaults to 12):
            Number of hidden layers in the RoBERTa encoder.
        num_attention_heads (:obj:`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the RoBERTa encoder.
        intermediate_size (:obj:`int`, optional, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the RoBERTa encoder.
        hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, "gelu", "relu", "swish" and "gelu_new" are supported.
        hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, optional, defaults to 514):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, optional, defaults to 1):
            The vocabulary size of the `token_type_ids` passed into :class:`~transformers.RobertaModel`.
        initializer_range (:obj:`float`, optional, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
    """

    pretrained_config_archive_map = ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "roberta"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
