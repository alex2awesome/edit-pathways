from modeling.layers_embedding import DocEmbedding
from modeling.layers_heads import PoissonRegressionSequenceHead, NormalRegressionSequenceHead, MultilabelSequenceHead
from modeling.models_base import BaseDiscriminator
from modeling.dataset_document import DocPredRow
from modeling.utils_lightning import LightningStepsDoc


class DocumentDiscriminator(LightningStepsDoc, BaseDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.doc_embeddings = DocEmbedding(expand=False, *args, **kwargs)

    def get_heads(self):
        if self.config.do_regression:
            if self.config.use_poisson_regression:
                self.head = PoissonRegressionSequenceHead(config=self.config, num_labels=self.config.num_labels)
            else:
                self.head = NormalRegressionSequenceHead(config=self.config, num_labels=self.config.num_labels)
        else:
            self.head = MultilabelSequenceHead(config=self.config, num_labels=self.config.num_labels)

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
        doc_embs = self.doc_embeddings(context_embs)

        # get losses
        loss, pred = self.head(doc_embs, label.labels)
        predictions = DocPredRow(preds=pred)
        return loss, predictions

