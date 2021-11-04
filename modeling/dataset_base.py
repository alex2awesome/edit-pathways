import torch
import torch.optim
import torch.utils.data as data
import pytorch_lightning as pl
import os
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer
import pandas as pd

from modeling.utils_general import reformat_model_path


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
        self.config = kwargs.get('config')
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

    def process_refactor(self, label):
        if self.do_regression:
            return label
        else:
            output = {'refactor up': 0, 'refactor unchanged': 0, 'refactor down': 0}
            if label == 0:
                output['refactor unchanged'] = 1
            elif label > 0:
                output['refactor up'] = 1
            elif label < 0:
                output['refactor down'] = 1
            return output

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
