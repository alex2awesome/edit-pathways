import torch
import torch.optim
import torch.utils.data as data
import pytorch_lightning as pl
import os
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer
import csv
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from modeling.utils_general import reformat_model_path, transpose_dict, _get_len


def _get_attention_mask(x, max_length_seq):
    max_len = max(map(lambda y: _get_len(y), x))
    max_len = min(max_len, max_length_seq)
    attention_masks = []
    for x_i in x:
        input_len = _get_len(x_i)
        if input_len < max_length_seq:
            mask = torch.cat((torch.ones(input_len), torch.zeros(max_len - input_len)))
        else:
            mask = torch.ones(max_length_seq)
        attention_masks.append(mask)
    return torch.stack(attention_masks)


class Dataset(data.Dataset):
    def __init__(self):
        """Reads source and target sequences from txt files."""
        self.data_rows = []

    def add_document(self, doc):
        self.data_rows.append(doc)

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.data_rows[index]


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data_fp = kwargs.get('data_fp')
        self.add_eos_token = (kwargs.get('model_type') == "gpt2")
        self.max_length_seq = kwargs.get('max_length_seq')
        self.do_regression = kwargs.get('do_regression')
        self.max_num_sentences = kwargs.get('max_num_sentences')
        self.batch_size = kwargs.get('batch_size')
        self.num_cpus = kwargs.get('num_cpus')
        self.split_type = kwargs.get('split_type')
        self.load_tokenizer(
            model_type=kwargs.get('model_type'), pretrained_model_path=kwargs.get('pretrained_model_path')
        )

        self.dataset = Dataset()


    def load_tokenizer(self, model_type, pretrained_model_path):
        if model_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
        elif model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
        elif model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(reformat_model_path(pretrained_model_path))
        else:
            print('Model path not in {bert, roberta, gpt2}.')

    def prepare_data(self):
        """
        Checks if the data path exists.

        Occurs only on the master GPU.
        """
        if not os.path.exists(self.data_fp):
            raise FileNotFoundError('Data files... make sure to download them from S3!')

    def process_sentence(self, text):
        text = text if pd.notnull(text) else ''
        seq = self.tokenizer.encode(text)
        if self.add_eos_token:
            seq.append(self.tokenizer.eos_token_id)
        seq = torch.tensor(seq, dtype=torch.long)
        seq = seq[:self.max_length_seq]
        return seq

    def process_label(self, label):
        if self.do_regression:
            return label
        else:
            if label not in [0, 1]:
                if abs(label) > 0:
                    return 1
                else:
                    return 0
            return label

    def setup(self, stage=None):
        """
            Download and split the dataset before training/testing.
            For Nonsequential datasets, this just splits on the sentences.
            For Sequential datasets (which are nested lists of sentences), this splits on the documents.

            Occurs on every GPU.
        """
        if stage in ('fit', None):
            d = self.get_dataset()
            # split randomly
            train_size = int(0.95 * len(d))
            test_size = len(d) - train_size
            self.train_dataset, self.test_dataset = torch.utils.data.random_split(d, [train_size, test_size])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_cpus
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_cpus
        )


class SentenceEditsModule(BaseDataModule):
    def process_document(self, doc_df):
        doc_df = doc_df.sort_values('sent_idx')
        sents = doc_df['sentence'].apply(self.process_sentence).tolist()
        labels_dict = {}
        labels_dict['num_add_after'] = doc_df['add_below_label'].apply(self.process_label).tolist()
        labels_dict['num_add_before'] = doc_df['add_above_label'].apply(self.process_label).tolist()
        labels_dict['refactor_distance'] = doc_df['refactored_label'].apply(self.process_label).tolist()
        labels_dict['is_deleted'] = doc_df['deleted_label'].tolist()
        labels_dict['is_edited'] = doc_df['edited_label'].tolist()
        labels_dict['is_unchanged'] = doc_df['unchanged_label'].tolist()

        data_row = SentenceDataRow(
            sent_idx=doc_df['sent_idx'].tolist(),
            sentences=sents,
            labels_dict=labels_dict,
            max_length_seq=self.max_length_seq
        )
        self.dataset.add_document(data_row)

    def get_dataset(self):
        """
        Read in csv with the fields:
            * entry_id
            * version
            * sent_idx
            * sentence
            * deleted_label
            * add_above_label
            * add_below_label
            * edited_label
            * unchanged_label
            * refactored_label

        Returns Dataset
        """
        input_data = pd.read_csv(self.data_fp)
        input_data['sentence'] = input_data['sentence'].fillna('')
        input_data.groupby(['entry_id', 'version']).apply(self.process_document)
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

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects the batch size to be 1.
        Expects dataset[i]['sentences'] to be a list of sentences and other fields (eg. dataset[i]['is_deleted']) to be a list of labels.
        Returns tensors X_batch, y_batch
        """
        data_rows = list(map(lambda x: x.collate(), dataset))
        label_rows = list(map(lambda x: x.labels, data_rows))
        # label_batch = SentenceLabelBatch(label_rows=label_rows)
        return {
            'input_ids': list(map(lambda x: x.sentence_batch, data_rows)),
            'attention_mask': list(map(lambda x: x.sentence_attention, data_rows)),
            'labels': label_rows
        }


class SentenceLabelRow():
    def __init__(self, labels_dict):
        self.num_add_before = labels_dict['num_add_before']
        self.num_add_after = labels_dict['num_add_after']
        self.refactor_distance = labels_dict['refactor_distance']
        self.sentence_operations_list = [
            labels_dict['is_deleted'],
            labels_dict['is_edited'],
            labels_dict['is_unchanged']
        ]

    def collate(self):
        self.num_add_before = torch.tensor(self.num_add_before, dtype=torch.long)
        self.num_add_after = torch.tensor(self.num_add_after, dtype=torch.long)
        self.refactor_distance = torch.tensor(self.refactor_distance, dtype=torch.long)
        self.sentence_operations_matrix = torch.tensor(self.sentence_operations_list, dtype=torch.long).T
        try:
            self.sentence_operations = torch.where(self.sentence_operations_matrix == 1)[1]
        except:
            print('FAILED:')
            print(self.sentence_operations_matrix)
        return self

    def to(self, device):
        self.num_add_before = self.num_add_before.to(device)
        self.num_add_after = self.num_add_after.to(device)
        self.refactor_distance = self.refactor_distance.to(device)
        self.sentence_operations = self.sentence_operations.to(device)
        return self

    @property
    def deleted(self):
        return (self.sentence_operations == 0).to(int)

    @property
    def edited(self):
        return (self.sentence_operations == 1).to(int)

    @property
    def unchanged(self):
        return (self.sentence_operations == 2).to(int)


class SentenceLabelBatch():
    def __init__(self, label_rows=None):
        # we might have an .add_label_row method so label_rows doesn't always have to passed in
        assert label_rows is not None
        self.labels = label_rows
        self.finalize()

    def finalize(self):
        self.num_add_before = torch.cat(list(map(lambda x: x.num_add_before, self.labels)))
        self.num_add_after = torch.cat(list(map(lambda x: x.num_add_after, self.labels)))
        self.refactor_distance = torch.cat(list(map(lambda x: x.refactor_distance, self.labels)))
        self.sentence_operations = torch.cat(list(map(lambda x: x.sentence_operations, self.labels)))

    def __getitem__(self, item):
        return self.labels[item]

    @property
    def deleted(self):
        return (self.sentence_operations == 0).to(int)

    @property
    def edited(self):
        return (self.sentence_operations == 1).to(int)

    @property
    def unchanged(self):
        return (self.sentence_operations == 2).to(int)

    def to(self, device):
        self.num_add_before = self.num_add_before.to(device)
        self.num_add_after = self.num_add_after.to(device)
        self.refactor_distance = self.refactor_distance.to(device)
        self.sentence_operations = self.sentence_operations.to(device)
        for label in self.labels:
            label.to(device)
        return self


class SentenceDataRow():
    def __init__(self, sent_idx, sentences, labels_dict, max_length_seq):
        self.sent_idx = sent_idx
        self.sentences = sentences
        self.max_length_seq = max_length_seq
        self.labels = SentenceLabelRow(labels_dict)

    def collate(self):
        self.sentence_batch = pad_sequence(self.sentences, batch_first=True)[:, :self.max_length_seq]
        self.sentence_attention = _get_attention_mask(self.sentences, self.max_length_seq)
        self.labels.collate()
        return self


class SentencePredBatch():
    def __init__(self):
        self.sent_ops = []
        self.sent_ops_lls = []
        self.add_before = []
        self.add_after = []
        self.refactored = []

    def add_row_to_batch(self, sentence_row):
        y_pred_sent_op = sentence_row.pred_sent_ops.argmax(dim=1)
        self.sent_ops.append(y_pred_sent_op)
        self.sent_ops_lls.append(sentence_row.pred_sent_ops)
        self.add_before.append(sentence_row.pred_added_before)
        self.add_after.append(sentence_row.pred_added_after)
        self.refactored.append(sentence_row.pred_refactored)

    def finalize(self):
        self.sent_ops = torch.cat(self.sent_ops)
        self.sent_ops_lls = torch.cat(self.sent_ops_lls)
        self.add_before = torch.cat(self.add_before).squeeze()
        self.add_after = torch.cat(self.add_after).squeeze()
        self.refactored = torch.cat(self.refactored).squeeze()

    @property
    def deleted(self):
        return (self.sent_ops == 0).to(float)

    @property
    def edited(self):
        return (self.sent_ops == 1).to(float)

    @property
    def unchanged(self):
        return (self.sent_ops == 2).to(float)



class SentencePredRow():
    def __init__(self, pred_added_after, pred_added_before, pred_refactored, pred_sent_ops):
        self.pred_added_after = pred_added_after
        self.pred_added_before = pred_added_before
        self.pred_refactored = pred_refactored
        self.pred_sent_ops = pred_sent_ops.argmax(axis=1)

    @property
    def deleted(self):
        return (self.pred_sent_ops == 0).to(float)

    @property
    def edited(self):
        return (self.pred_sent_ops == 1).to(float)

    @property
    def unchanged(self):
        return (self.pred_sent_ops == 2).to(float)
