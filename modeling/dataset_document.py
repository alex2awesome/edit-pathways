import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from modeling.dataset_base import BaseDataModule
from modeling.utils_general import _get_attention_mask


class DocumentEditsModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_label_order = None

    def process_document(self, doc_s):
        sents = doc_s['sentences'].split('<SENT>')
        sents = list(map(self.process_sentence, sents))
        labels = (
            doc_s
              .drop('sentences')
              .astype(int)
        )
        if self.class_label_order is None:
            self.class_label_order = labels.index.tolist()
        else:
            labels = labels[self.class_label_order]

        data_row = DocDataRow(
            sentences=sents,
            labels=labels,
            max_length_seq=self.max_length_seq,
            do_regression=self.do_regression
        )
        self.dataset.add_document(data_row)

    def get_dataset(self):
        """
        Read in csv where each row represents a document, with the following fields:
            * sentences
            * 'add_above_label, 0/n, ...'
            * 'add_below_label, 0/n...'
            * deleted_label, 0/n'...
            * 'edited_label, 0/n...'
            * 'not refactored, 0/n...'
            * 'refactored down, 0/n'...
            * 'unchanged_label, 0/n'
        Returns Dataset
        """
        input_data = pd.read_csv(self.data_fp).drop(['source', 'entry_id', 'version'], axis=1)
        if self.config.local:
            input_data = input_data.head(100)
        input_data['sentences'] = input_data['sentences'].fillna('')
        input_data.apply(self.process_document, axis=1)
        return self.dataset

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch['attention_mask'] = list(map(lambda x: x.to(device), batch['attention_mask']))
        batch['input_ids'] = list(map(lambda x: x.to(device), batch['input_ids']))
        labels = batch['labels']
        if isinstance(labels, list):
            batch['labels'] = list(map(lambda x: x.to(device), labels))
        else:
            batch['labels'] = labels.to(device)
        return batch

    def collate_fn(self, data_rows):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects the batch size to be 1.
        Expects dataset[i]['sentences'] to be a list of sentences and other fields (eg. dataset[i]['is_deleted']) to be a list of labels.
        Returns tensors X_batch, y_batch
        """
        label_rows = list(map(lambda x: x.labels, data_rows))
        return {
            'input_ids': list(map(lambda x: x.sentence_batch, data_rows)),
            'attention_mask': list(map(lambda x: x.sentence_attention, data_rows)),
            'labels': label_rows
        }


class DocDataRow():
    def __init__(self, sentences, labels, max_length_seq, do_regression):
        self.sentences = sentences
        self.max_length_seq = max_length_seq
        self.labels = DocLabelRow(labels, do_regression)
        self.sentence_batch = pad_sequence(self.sentences, batch_first=True)[:, :self.max_length_seq]
        self.sentence_attention = _get_attention_mask(self.sentences, self.max_length_seq)


class DocLabelRow():
    def __init__(self, labels, do_regression):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vals = labels.values.tolist()
        if not do_regression:
            self.labels = (self.labels > 0).to(torch.long)

    def to(self, device):
        self.labels = self.labels.to(device)
        return self


class DocPredRow():
    def __init__(self, preds):
        self.preds = preds.squeeze()