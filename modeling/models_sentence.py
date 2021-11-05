import torch
from torch import nn
from torch.autograd import Variable

from modeling.layers_embedding import PosEmbedding
from modeling.layers_heads import PoissonRegressionSequenceHead, NormalRegressionSequenceHead, BinaryClassSequenceHead, \
    MultiClassSequenceHead
from modeling.models_base import BaseDiscriminator
from modeling.dataset_sentence import SentencePredRow
from modeling.utils_lightning import LightningStepsSentence
from modeling.utils_mixer import Mixer


class SentenceDiscriminator(LightningStepsSentence, BaseDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_embeddings = PosEmbedding(*args, **kwargs)

    def predict_one_doc(self, input_ids, label, attention_mask=None, sequence_lens=None):
        """
        Parameters:
             * `input_ids`: one document tokens (list of sentences. Each sentence is a list of ints.)
             * `labels`: list of y_preds [optional].
             * `attention`: list

        """
        sent_embs = self.sentence_embeddings.get_sentence_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_lens=sequence_lens,
        )
        if self.config.use_contextual_layers:
            sent_embs = self.sentence_emb_context(sent_embs)
        else:
            sent_embs = self.resize_layer(sent_embs)

        pos_embs = self.pos_embeddings(sent_embs)
        doc_embs = self.doc_embeddings(sent_embs)

        # get losses
        losses = []
        preds = {}
        losses, preds = self.run(sent_embs, label, pos_embs, doc_embs, losses, preds)
        predictions = SentencePredRow(**preds, use_deepspeed=self.config.use_deepspeed)
        losses = Variable(torch.tensor(losses), requires_grad=True)
        if losses.shape == self.loss_weighting.shape:
            loss = self.loss_weighting.dot(losses)
        else:
            loss = losses.sum()
        loss = Variable(loss, requires_grad=True)
        return loss, predictions


class Addition(nn.Module):
    def get_heads(self):
        super().get_heads()
        if self.config.do_regression:
            if self.config.use_poisson_regression:
                head = PoissonRegressionSequenceHead
            else:
                head = NormalRegressionSequenceHead
        else:
            head = BinaryClassSequenceHead

        self.add_before_head = head(config=self.config)
        self.add_after_head = head(config=self.config)

    def run(self, context_embs, label, pos_embs, doc_embs, losses, preds):
        super().run(context_embs, label, pos_embs, doc_embs, losses, preds)
        loss, pred = self.add_after_head(context_embs, label.num_add_after, pos_embs, doc_embs)
        losses.append(loss)
        preds['pred_added_after'] = pred
        loss, pred = self.add_before_head(context_embs, label.num_add_before, pos_embs, doc_embs)
        losses.append(loss)
        preds['pred_added_before'] = pred
        return losses, preds


class SentenceOps(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_heads(self):
        super().get_heads()
        self.sentence_operation_head = MultiClassSequenceHead(config=self.config)

    def run(self, context_embs, label, pos_embs, doc_embs, losses, preds):
        super().run(context_embs, label, pos_embs, doc_embs, losses, preds)
        loss, pred = self.sentence_operation_head(context_embs, label.sentence_operations, pos_embs, doc_embs)
        losses.append(loss)
        preds['pred_sent_ops'] = pred
        return losses, preds


class RefactorRegression(nn.Module):
    def get_heads(self):
        super().get_heads()
        self.refactor_head = NormalRegressionSequenceHead(config=self.config)

    def run(self, context_embs, label, pos_embs, doc_embs, losses, preds):
        super().run(context_embs, label, pos_embs, doc_embs, losses, preds)
        loss, pred = self.refactor_head(context_embs, label.refactor_distance, pos_embs, doc_embs)
        losses.append(loss)
        preds['pred_refactored'] = pred
        return losses, preds


class RefactorClassification(nn.Module):
    def get_heads(self):
        super().get_heads()
        self.refactor_head = MultiClassSequenceHead(config=self.config)

    def run(self, context_embs, label, pos_embs, doc_embs, losses, preds):
        super().run(context_embs, label, pos_embs, doc_embs, losses, preds)
        loss, pred = self.refactor_head(context_embs, label.refactor_ops, pos_embs, doc_embs)
        losses.append(loss)
        preds['pred_refactored_ops'] = pred
        return losses, preds


class ModelHeadBlank(nn.Module):
    def get_heads(self):
        pass

    def run(self, *args, **kwargs):
        pass


def get_sentence_model(config):
    model_mixins = []
    if config.do_addition:
        model_mixins.append(Addition)
    if config.do_refactor:
        if config.do_regression:
            model_mixins.append(RefactorRegression)
        else:
            model_mixins.append(RefactorClassification)
    if config.do_operations:
        model_mixins.append(SentenceOps)
    model_mixins.append(ModelHeadBlank)
    model_mixins.append(SentenceDiscriminator)
    return Mixer(config=config, mixin=model_mixins)