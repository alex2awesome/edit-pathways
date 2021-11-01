from torch import nn
from transformers import AutoConfig, GPT2Model, RobertaModel, BertModel

from modeling.utils_general import get_config

class TransformerContext(nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        transformer_config = AutoConfig.from_pretrained(self.config.pretrained_cache_dir)
        orig_embed_size = transformer_config.hidden_size
        assert hasattr(self.config, 'num_sent_attn_heads') and hasattr(self.config, 'num_contextual_layers')
        super().__init__()
        # resize
        if orig_embed_size != self.config.hidden_dim:
            self.resize_layer = nn.Linear(orig_embed_size, self.config.hidden_dim, bias=False)
            self.do_resize = True
        else:
            self.do_resize = False
        # load transformer
        if self.config.model_type == 'gpt2':
            transformer_config.n_head = self.config.num_sent_attn_heads
            transformer_config.n_layer = self.config.num_contextual_layers
            transformer_config.n_embd = self.config.hidden_dim
            transformer_config.n_positions = self.config.max_num_sentences + 20
            transformer_config.n_ctx = transformer_config.n_positions
            self.sentence_transformer = GPT2Model(config=transformer_config)
        elif self.config.model_type == 'roberta':
            transformer_config.num_attention_heads = self.config.num_sent_attn_heads
            transformer_config.num_hidden_layers = self.config.num_contextual_layers
            transformer_config.hidden_size = self.config.hidden_dim
            transformer_config.max_position_embeddings = self.config.max_num_sentences + 20
            self.sentence_transformer = RobertaModel(config=transformer_config)
        elif self.config.model_type == 'bert':
            self.sentence_transformer = BertModel(config=transformer_config)

    def get_final_hidden_layer_size(self):
        return self.config.hidden_dim

    def forward(self, cls_embeddings):
        if self.do_resize: # pass vector through a linear layer to resize it
            cls_embeddings = self.resize_layer(cls_embeddings)
        if self.config.model_type == 'gpt2':
            contextualized_embeds, _ = self.sentence_transformer(inputs_embeds=cls_embeddings, return_dict=False)
        else:
            contextualized_embeds, _ = self.sentence_transformer(inputs_embeds=cls_embeddings.unsqueeze(0), return_dict=False)
            contextualized_embeds = contextualized_embeds.squeeze(dim=0)
        return contextualized_embeds


