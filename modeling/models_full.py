from modeling.layers_context import TransformerContext
from modeling.layers_embedding import SentenceEmbeddingsLayer, PosEmbedding, DocEmbedding
from torch import nn
from operator import mul
from modeling.utils_general import vec_or_nones, get_config, format_loss_weighting, SuperBlank
import torch
from modeling.layers_heads import (
    MultiClassSequenceHead,
    NormalRegressionSequenceHead,
    PoissonRegressionSequenceHead,
    BinaryClassSequenceHead
)
from modeling.utils_dataset import SentencePredBatch, SentencePredRow
from modeling.utils_lightning import LightningStepsBase, LightningOptimizer
from torch.autograd import Variable


class SentenceDiscriminator(LightningStepsBase, LightningOptimizer, SuperBlank, nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        super().__init__(config=self.config)
        self.sentence_embeddings = SentenceEmbeddingsLayer(*args, **kwargs)
        self.sentence_emb_context = TransformerContext(*args, **kwargs)
        self.doc_embeddings = DocEmbedding(*args, **kwargs)
        self.pos_embeddings = PosEmbedding(*args, **kwargs)
        self.loss_weighting = format_loss_weighting(self.config.loss_weighting)
        self.get_heads()

    def get_heads(self):
        self.sentence_operation_head = MultiClassSequenceHead(config=self.config)
        if self.config.do_regression:
            self.refactor_head = NormalRegressionSequenceHead(config=self.config)
            if self.config.use_poisson_regression:
                self.add_before_head = PoissonRegressionSequenceHead(config=self.config)
                self.add_after_head = PoissonRegressionSequenceHead(config=self.config)
            else:
                self.add_before_head = NormalRegressionSequenceHead(config=self.config)
                self.add_after_head = NormalRegressionSequenceHead(config=self.config)
        else:
            self.refactor_head = MultiClassSequenceHead(config=self.config)
            self.add_before_head = BinaryClassSequenceHead(config=self.config)
            self.add_after_head = BinaryClassSequenceHead(config=self.config)

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
        context_embs = self.sentence_emb_context(sent_embs)
        pos_embs = self.pos_embeddings(context_embs)
        doc_embs = self.doc_embeddings(context_embs)

        # get losses
        loss_added_afer, pred_added_after = self.add_after_head(context_embs, label.num_add_after, pos_embs, doc_embs)
        loss_added_before, pred_added_before = self.add_before_head(context_embs, label.num_add_before, pos_embs, doc_embs)
        pred_refactored, pred_refactored_ops = None, None
        if self.config.do_regression:
            loss_refactored, pred_refactored = self.refactor_head(context_embs, label.refactor_distance, pos_embs, doc_embs)
        else:
            loss_refactored, pred_refactored_ops = self.refactor_head(
                context_embs,
                label.refactor_ops,
                pos_embs,
                doc_embs
            )
        loss_sent_ops, pred_sent_ops = self.sentence_operation_head(context_embs, label.sentence_operations, pos_embs, doc_embs)

        predictions = SentencePredRow(
            pred_sent_ops=pred_sent_ops,
            pred_refactored=pred_refactored,
            pred_refactored_ops=pred_refactored_ops,
            pred_added_before=pred_added_before,
            pred_added_after=pred_added_after,
            use_deepspeed=self.config.use_deepspeed
        )

        losses = Variable(torch.tensor([loss_added_afer, loss_added_before, loss_refactored, loss_sent_ops]), requires_grad=True)
        loss = self.loss_weighting.dot(losses)
        loss = Variable(loss, requires_grad=True)
        return loss, predictions

    def forward(self, input_ids, labels, attention_mask=None, input_lens=None):
        """
        Step that's shared between training loop and validation loop. Contains sequence-specific processing,
        so we're keeping it in the child class.

        Parameters:
            * `input_ids`: list of docs, len(input_ids) = # batches (default = 1).
                Each item is a flat list of token-ids of length `num_toks_in_doc`.
            * `labels`: [optional] list of sentence-level labels of length batch_size.
                Each item contains tensor of labels length `num_sents_in_doc`.
            * `attention_mask`: [optional] list of attention matrices of length batch_size.
                Each item is a matrix of size `num_sents_in_doc` x `max_i[num tokens in sent i]`
            * `input_lens` [optional]: list of sentence-lengths of length `batch_size`.
                Each item is a tensor of length `num_sentences_in_doc`.


        Returns tuple of (loss, y_preds, y_trues)
         if labels is not None, else
         returns tuple of (None, y_preds, None)
        """
        if isinstance(input_ids, list):
            # batch is list of docs
            losses = []
            preds_batch = [] # SentencePredBatch()
            attention_mask = vec_or_nones(attention_mask, len(input_ids))
            input_lens = vec_or_nones(input_lens, len(input_ids))
            #
            for idx in range(len(input_ids)):
                # for X, y, a, s in zip(input_ids, labels, attention_mask, input_lens):
                X = input_ids[idx]
                if len(X.shape) == 0:
                    continue
                loss, preds = self.predict_one_doc(
                    X, label=labels[idx], attention_mask=attention_mask[idx], sequence_lens=input_lens[idx]
                )

                losses.append(loss)
                preds_batch.append(preds)# add_row_to_batch(sentence_row=preds)

            loss = sum(losses)
            # preds_batch.finalize()

            return loss, preds_batch
