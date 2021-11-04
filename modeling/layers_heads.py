import torch
from torch import nn


class SequenceHeadBase(nn.Module):
    """
    Takes in a tensor of sentence embeddings and a document embedding. Concatenates and performs a classification.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        ## final classification head
        self.config = config
        hidden_dim = config.hidden_dim
        self.num_addt_vectors = int(config.use_positional) + \
                                int(config.use_doc_emb) + \
                                int(config.doc_embed_arithmetic) # We add 2 vectors. However, use_doc_embed provides 1.
        # accounts for 1 of them.
        self.concatenated_hidden_dim = hidden_dim * (1 + self.num_addt_vectors)
        #
        if self.num_addt_vectors > 0:
            self.pre_pred = nn.Linear(self.concatenated_hidden_dim, hidden_dim)  # orig_code: config.embedding_dim)

    def _init_prediction_weights(self):
        if self.num_addt_vectors > 0:
            nn.init.xavier_uniform_(self.pre_pred.state_dict()['weight'])
            self.pre_pred.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.pred.state_dict()['weight'])
        self.pred.bias.data.fill_(0)

    def concat_vectors(self, sentence_embs, position_embeddings=None, doc_embedding=None):
        to_concat = [sentence_embs]
        if self.config.use_positional:
            to_concat += [position_embeddings]
        if self.config.use_doc_emb:
            if self.config.doc_embed_arithmetic:
                to_concat += [doc_embedding * sentence_embs, doc_embedding - sentence_embs]
            else:
                to_concat += [doc_embedding]
        return torch.cat(to_concat, 1)

    def calculate_loss(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss

    def forward(self, sentence_embs, labels, position_embeddings=None, doc_embedding=None):
        # loss
        pred = self.perform_prediction(sentence_embs, position_embeddings, doc_embedding)
        loss = self.calculate_loss(pred, labels)
        loss = torch.mean(loss)
        return loss, pred

    def perform_prediction(self, sentence_embs, position_embeddings=None, doc_embedding=None):
        concatted_embs = self.concat_vectors(sentence_embs, position_embeddings, doc_embedding)
        if self.num_addt_vectors > 0:
            concatted_embs = self.pre_pred(
                self.drop(torch.tanh(concatted_embs)))           ## pre_pred = (batch_size x hidden_dim * 2)
        return self.pred(self.drop(torch.tanh(concatted_embs)))  ## pred = ( batch_size x num_labels)


class BinaryClassSequenceHead(SequenceHeadBase):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.pred = nn.Linear(config.hidden_dim, 1)
        self.drop = nn.Dropout(config.dropout)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def calculate_loss(self, preds, labels):
        preds = preds.squeeze()
        labels = labels.float()
        loss = self.criterion(preds, labels)
        return loss


class MultilabelSequenceHead(SequenceHeadBase):
    def __init__(self, num_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred = nn.Linear(self.config.hidden_dim, num_labels)
        self.drop = nn.Dropout(self.config.dropout)
        self.criterion = nn.BCEWithLogitsLoss()

    def calculate_loss(self, preds, labels):
        preds = preds.reshape(labels.shape)
        labels = labels.float()
        return super().calculate_loss(preds, labels)


class MultiClassSequenceHead(SequenceHeadBase):
    def __init__(self, num_labels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred = nn.Linear(self.config.hidden_dim, num_labels)
        self.drop = nn.Dropout(self.config.dropout)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self._init_prediction_weights()


class NormalRegressionSequenceHead(SequenceHeadBase):
    def __init__(self, num_labels=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred = nn.Linear(self.config.hidden_dim, num_labels)
        self.drop = nn.Dropout(self.config.dropout)
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, sentence_embs, labels, position_embeddings=None, doc_embedding=None):
        # loss
        pred = self.perform_prediction(sentence_embs, position_embeddings, doc_embedding)
        pred = pred.reshape(labels.shape)
        loss = self.calculate_loss(pred, labels)
        loss = torch.mean(loss)
        return loss, pred


class PoissonRegressionSequenceHead(NormalRegressionSequenceHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp = torch.exp
        self.criterion = torch.nn.PoissonNLLLoss(reduction='none')

    def perform_prediction(self, sentence_embs, position_embeddings=None, doc_embedding=None):
        pred = super().perform_prediction(sentence_embs, position_embeddings, doc_embedding)
        return torch.exp(pred)
    

class BetaRegressionSequenceHead(NormalRegressionSequenceHead):
    def __init__(self, *args, **kwargs):
        from pytorch_forecasting.metrics import BetaDistributionLoss
        super().__init__(*args, **kwargs)
        self.exp = torch.exp
        self.criterion = BetaDistributionLoss(reduction='none')
