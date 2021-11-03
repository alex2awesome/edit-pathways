import torch
import torch.nn as nn

class AdditiveSelfAttention(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, input_dim)
        self.ws2 = nn.Linear(input_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.ws2.state_dict()['weight'])
        self.ws2.bias.data.fill_(0)

    def forward(self, sent_embeds, context_mask=None):
        ## get sentence encoding using additive attention (appears to be based on Bahdanau 2015) where:
        ##     score(s_t, h_i) = v_a^T tanh(W_a * [s_t; h_i]),
        ## here, s_t, h_i = word embeddings
        ## align(emb) = softmax(score(Bi-LSTM(word_emb)))
        # word_embs: shape = (num sentences in curr batch * max_len * embedding_dim)   # for word-attention:
        #     where embedding_dim = hidden_dim * 2                                     # -------------------------------------
        self_attention = torch.tanh(self.ws1(self.drop(sent_embeds)))                  # self attention : (num sentences in curr batch x max_len x (hidden_dim * 2))
        self_attention = self.ws2(self.drop(self_attention)).squeeze(-1)               # self_attention : (num_sentences in curr batch x max_len)
        if context_mask is not None:
            self_attention = self_attention + -10000 * (context_mask == 0).float()     # self_attention : (num_sentences in curr batch x max_len)
        self_attention = self.softmax(self_attention)                                  # self_attention : (num_sentences in curr batch x max_len)
        return self_attention


class SentenceLevelSelfAttention(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.self_attention = AdditiveSelfAttention(input_dim=input_dim, dropout=config.dropout)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, contexualized_word_embs, context_mask):
        self_attention = self.self_attention(contexualized_word_embs, context_mask)              #      sent_encoding: (# sents in batch x hidden_dim )
        sent_encoding = torch.sum(contexualized_word_embs * self_attention.unsqueeze(-1), dim=1)
        return self.drop(sent_encoding)


class DocLevelSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = AdditiveSelfAttention(input_dim=config.hidden_dim, dropout=config.dropout)

    def forward(self, sent_encoding):
        ## get document embedding?
        ## inner_pred: shape = 1 x batch_size x (hidden_dim * 2)
        ## sent_encoding: shape= batch_size x (hidden_dim * 2)
        sent_encoding = sent_encoding.unsqueeze(0)
        self_attention = self.self_attention(sent_encoding)                   # self_attention = 1 x batch_size
        doc_encoding = torch.matmul(self_attention.squeeze(), sent_encoding)  # doc_encoding   = 1 x (hidden_dim * 2)

        ## reshape
        sent_encoding = sent_encoding.squeeze()                               # inner_pred = batch_size x (hidden_dim * 2)
        doc_encoding = doc_encoding.expand(sent_encoding.size())              #  doc_encoding = batch_size x (hidden_dim * 2)
        return doc_encoding