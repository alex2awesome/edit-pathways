from torch import nn
from transformers import AutoConfig
import torch
from transformers import GPT2LMHeadModel, BertModel, RobertaModel
from modeling.utils_general import get_config, format_layer_freezes, freeze_all_params, reshape_and_pad_sequence
from modeling.layers_attention import SentenceLevelSelfAttention, DocLevelSelfAttention

class SentenceEmbeddingsLayer(nn.Module):
    """
    Main Discourse generator_pplm class:

        Base Document encoder: RoBERTa
        Head: Bi-LSTM or CRF
    """
    def __init__(self, config=None, loading_from_checkpoint=False, *args, **kwargs):
        # setup configs
        self.config = get_config(config=config, kwargs=kwargs)
        self.loading_from_checkpoint = loading_from_checkpoint
        super().__init__()

        # get encoder
        self.get_pretrained_model()

        # freeze layers
        self.freeze_encoder_layers()

        # setup dropout
        self.dropout = nn.Dropout(self.config.dropout)

        #
        self.additive_attention = SentenceLevelSelfAttention(config=self.config, input_dim=self.embed_size)

    def get_pretrained_model(self):
        # get pretrained model
        if self.config.model_type == "gpt2":
            transformer_config = AutoConfig.from_pretrained(self.config.pretrained_cache_dir)
            transformer_config.n_ctx = transformer_config.n_positions = self.config.max_num_word_positions
            self.embed_size = transformer_config.hidden_size
            ######### if loading from a checkpoint
            # Initialize the model structure - pytorch_lightning will call `load_state_dict()`.
            # This is lighter-weight than loading the pretrained model just to overwrite the weights.
            if self.loading_from_checkpoint:
                self.encoder_model = GPT2LMHeadModel(config=transformer_config)
            else:
                self.encoder_model = GPT2LMHeadModel.from_pretrained(self.config.pretrained_cache_dir, config=transformer_config)
            ##
        elif self.config.model_type == "bert":
            self.encoder_model = BertModel.from_pretrained(self.config.pretrained_cache_dir)
            self.embed_size = self.encoder_model.config.hidden_size
        elif self.config.model_type == 'roberta':
            self.encoder_model = RobertaModel.from_pretrained(self.config.pretrained_cache_dir)
            self.embed_size = self.encoder_model.config.hidden_size
        else:
            raise ValueError(
                "{} model not yet supported".format(self.config.model_type)
            )

    def freeze_encoder_layers(self):
        # freeze whole transformer
        if self.config.freeze_transformer:
            freeze_all_params(self.encoder_model)

        # freeze embedding layer
        if self.config.freeze_embedding_layer:
            if self.config.model_type == 'gpt2':
                freeze_all_params(self.encoder_model.transformer.wte)
            else:
                freeze_all_params(self.encoder_model.embeddings)

        # freeze encoding layers
        if self.config.freeze_encoder_layers:
            layers_to_freeze = format_layer_freezes(self.config.freeze_encoder_layers)
            for layer in layers_to_freeze:
                if self.config.model_type == 'gpt2':
                    freeze_all_params(self.encoder_model.transformer.h[layer])
                else:
                    freeze_all_params(self.encoder_model.encoder.layer[layer])

    def _get_word_embeddings(self, input_ids=None, attention_mask=None):
        if hasattr(self.encoder_model, 'transformer'):
            model = self.encoder_model.transformer
        else:
            model = self.encoder_model

        hidden, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return hidden

    def get_sentence_embedding(self, input_ids, attention_mask, sequence_lens=None):
        """
        Helper method to calculate sentence embeddings for text.

        Parameters:
            * input_ids: normally, this is a matrix of size (len doc X max len sents) (unless sequence_lens is passed).
            * attention_mask: matrix of size (len doc X max len sents) with zeros to represent input_id padding.
            * sequence_lens: if passed, assume that input_ids is of shape (num docs X total doc len).
            * get_last: if true, return the sentence embedding of the last sentence in the doc.
            * get_word_embs: if true, return the sequence of hidden states.
        """
        hidden = self._get_word_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask if sequence_lens is None else None,
        )

        if sequence_lens is not None:
            hidden = reshape_and_pad_sequence(hidden, sequence_lens)

        return self.additive_attention(hidden, attention_mask)


class DocEmbedding(nn.Module):
    def __init__(self, config, expand=True):
        super().__init__()
        self.doc_attention = DocLevelSelfAttention(config, expand)  ## we find DocLevelSelfAttention performs the same/better than DocLevelAttention

    def forward(self, sentence_embeddings):
        return self.doc_attention(sentence_embeddings)


class PosEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.config = config
        if not self.config.sinusoidal_embeddings:
            self.max_position_embs = nn.Parameter(torch.tensor(config.max_position_embeddings), requires_grad=False)
            self.default_max_pos = nn.Parameter(torch.tensor(self.max_position_embs - 1), requires_grad=False)
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, hidden_dim)
        else:
            from fairseq.modules import SinusoidalPositionalEmbedding
            self.position_embeddings = SinusoidalPositionalEmbedding(hidden_dim, self.config.pad_id, config.max_position_embeddings)

    def forward(self, hidden_embs):
        # get position embeddings
        if not self.config.sinusoidal_embeddings:
            position_ids = torch.arange(len(hidden_embs), dtype=torch.long, device=hidden_embs.device)
            position_ids = position_ids.where(position_ids < self.max_position_embs, self.default_max_pos) ## assign long sequences the same embedding
            position_embeddings = self.position_embeddings(position_ids)
        else:
            position_embeddings = self.position_embeddings(hidden_embs[:, [0]]).squeeze()

        return position_embeddings

