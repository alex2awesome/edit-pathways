from modeling.layers_context import TransformerContext
from modeling.layers_embedding import SentenceEmbeddingsLayer, DocEmbedding
from torch import nn
from modeling.utils_general import vec_or_nones, get_config, format_loss_weighting, SuperBlank
from modeling.utils_lightning import LightningOptimizer


class BaseDiscriminator(LightningOptimizer, SuperBlank, nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        super().__init__(*args, **kwargs)
        self.sentence_embeddings = SentenceEmbeddingsLayer(*args, **kwargs)
        if self.config.use_contextual_layers:
            self.sentence_emb_context = TransformerContext(*args, **kwargs)
        else:
            self.resize_layer = nn.Linear(self.config.embedding_dim, self.config.hidden_dim)
        self.doc_embeddings = DocEmbedding(*args, **kwargs)
        self.loss_weighting = format_loss_weighting(self.config.loss_weighting)
        self.get_heads()

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


