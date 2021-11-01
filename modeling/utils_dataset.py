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


class SentenceDataRow():
    def __init__(self, sent_idx, sentences, num_add_before, num_add_after, refactor_distance, is_deleted, is_edited, is_unchanged):
        self.sent_idx = sent_idx
        self.sentences = sentences
        self.num_add_before = num_add_before
        self.num_add_after = num_add_after
        self.refactor_distance = refactor_distance
        self.sentence_operations = [is_deleted, is_edited, is_unchanged]


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
        self.do_regression = kwargs.get('regression_type') != 'classification'
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
            train_size = int(0.9 * len(d))
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

    def _get_attention_mask(self, x):
        return _get_attention_mask(x, self.max_length_seq)


class SentenceEditsModule(BaseDataModule):
    def process_document(self, doc_df):
        doc_df = doc_df.sort_values('sent_idx')
        sents = doc_df['sentence'].apply(self.process_sentence).tolist()
        add_after = doc_df['add_above_label'].apply(self.process_label).tolist()
        add_before = doc_df['add_before_label'].apply(self.process_label).tolist()
        refactor_distance = doc_df['refactored_label'].apply(self.process_label).tolist()

        data_row = SentenceDataRow(sent_idx=doc_df['sent_idx'].tolist(),
                                   sentences=sents,
                                   num_add_before=add_before,
                                   num_add_after=add_after,
                                   refactor_distance=refactor_distance,
                                   is_deleted=doc_df['deleted_label'].tolist(),
                                   is_edited=doc_df['edited_label'].tolist(),
                                   is_unchanged=doc_df['unchanged_label'].tolist(),
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
        input_data.groupby(['entry_id', 'version']).apply(self.process_document)
        return self.dataset

    def collate_fn(self, dataset):
        """
        Takes in an instance of Torch Dataset (or a subclassed instance).
        Expects the batch size to be 1.
        Expects dataset[i]['sentences'] to be a list of sentences and other fields (eg. dataset[i]['is_deleted']) to be a list of labels.
        Returns tensors X_batch, y_batch
        """
        assert len(dataset) == 1
        columns = transpose_dict(dataset[0].__dict__)
        sentence_batch = pad_sequence(columns["sentences"], batch_first=True)
        sentence_attention = self._get_attention_mask(columns["sentences"])
        num_add_before = torch.tensor(columns['num_add_before'], dtype=torch.long)
        num_add_after = torch.tensor(columns['num_add_after'], dtype=torch.long)
        refactor_distance = torch.tensor(columns['refactor_distance'], dtype=torch.long)
        sentence_operations = torch.tensor(columns['sentence_operations'], dtype=torch.long)

        return {
            'input_ids': sentence_batch,
            'attention_mask': sentence_attention,
            'labels_num_added_after': num_add_after,
            'labels_num_added_before': num_add_before,
            'labels_refactor_distance': refactor_distance,
            'labels_sentence_ops': sentence_operations,
        }