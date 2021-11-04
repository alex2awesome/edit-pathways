import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from modeling.dataset_base import BaseDataModule
from modeling.utils_general import _get_attention_mask
from modeling.utils_mixer import Mixer


class SentenceDataRow():
    def __init__(self, sent_idx, sentences, labels_dict, config):
        self.sent_idx = sent_idx
        self.sentences = sentences
        self.labels = get_label_row_lame(config, labels_dict)
        self.sentence_batch = pad_sequence(self.sentences, batch_first=True)[:, :config.max_length_seq]
        self.sentence_attention = _get_attention_mask(self.sentences, config.max_length_seq)


class SentencePredRow():
    def __init__(
        self,
        pred_added_after=None,
        pred_added_before=None,
        pred_sent_ops=None,
        pred_refactored=None,
        pred_refactored_ops=None,
        use_deepspeed=False
    ):
        data_type = float if not use_deepspeed else torch.half
        self.pred_added_after = pred_added_after
        self.pred_added_before = pred_added_before
        self.pred_refactored = pred_refactored
        if pred_refactored_ops is not None:
            self.pred_refactored_ops = pred_refactored_ops.argmax(axis=1)
            self.pred_ref_up = (self.pred_refactored_ops == 0).to(data_type)
            self.pred_ref_un = (self.pred_refactored_ops == 1).to(data_type)
            self.pred_ref_down = (self.pred_refactored_ops == 2).to(data_type)
        if pred_sent_ops is not None:
            self.pred_sent_ops = pred_sent_ops.argmax(axis=1)
            self.deleted = (self.pred_sent_ops == 0).to(data_type)
            self.edited = (self.pred_sent_ops == 1).to(data_type)
            self.unchanged = (self.pred_sent_ops == 2).to(data_type)


class SentenceEditsModule(BaseDataModule):
    def process_document(self, doc_df):
        doc_df = doc_df.sort_values('sent_idx')
        sents = doc_df['sentence'].apply(self.process_sentence).tolist()
        labels_dict = {}
        labels_dict['num_add_after'] = doc_df['add_below_label'].apply(self.process_label).tolist()
        labels_dict['num_add_before'] = doc_df['add_above_label'].apply(self.process_label).tolist()
        if not self.do_regression:
            refactor = doc_df['refactored_label'].apply(lambda x: pd.Series(self.process_refactor(x)))
            labels_dict['refactor_up'] = refactor['refactor up'].tolist()
            labels_dict['refactor_unchanged'] = refactor['refactor unchanged'].tolist()
            labels_dict['refactor_down'] = refactor['refactor down'].tolist()
        else:
            labels_dict['refactor_distance'] = doc_df['refactored_label'].tolist()

        labels_dict['is_deleted'] = doc_df['deleted_label'].tolist()
        labels_dict['is_edited'] = doc_df['edited_label'].tolist()
        labels_dict['is_unchanged'] = doc_df['unchanged_label'].tolist()

        data_row = SentenceDataRow(
            sent_idx=doc_df['sent_idx'].tolist(),
            sentences=sents,
            labels_dict=labels_dict,
            config=self.config
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
        to_group = ['entry_id', 'version']
        if 'source' in input_data.columns:
            to_group.append('source')
        input_data.groupby(to_group).apply(self.process_document)
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


class SentenceLabelRowAddition():
    def __init__(self, labels_dict):
        super().__init__(labels_dict)
        self.num_add_before = torch.tensor(labels_dict['num_add_before'], dtype=torch.long)
        self.num_add_after = torch.tensor(labels_dict['num_add_after'], dtype=torch.long)

    def to(self, device):
        super().to(device)
        self.num_add_before = self.num_add_before.to(device)
        self.num_add_after = self.num_add_after.to(device)
        return self


class SentenceLabelRowRefactorRegression():
    def __init__(self, labels_dict):
        super().__init__(labels_dict)
        self.refactor_distance = torch.tensor(labels_dict['refactor_distance'], dtype=torch.long)

    def to(self, device):
        super().to(device)
        self.refactor_distance = self.refactor_distance.to(device)
        return self


class SentenceLabelRowRefactorClassification():
    def __init__(self, labels_dict):
        super().__init__(labels_dict)
        refactor_ops_list = [
            labels_dict['refactor_up'],
            labels_dict['refactor_unchanged'],
            labels_dict['refactor_down'],
        ]
        refactor_ops_matrix = torch.tensor(refactor_ops_list, dtype=torch.long).T
        self.refactor_ops = torch.where(refactor_ops_matrix == 1)[1]
        self.refactor_up = (self.refactor_ops == 0).to(int)
        self.refactor_unchanged = (self.refactor_ops == 1).to(int)
        self.refactor_down = (self.refactor_ops == 2).to(int)

    def to(self, device):
        super().to(device)
        self.refactor_up = self.refactor_up.to(device)
        self.refactor_down = self.refactor_down.to(device)
        self.refactor_unchanged = self.refactor_unchanged.to(device)
        self.refactor_ops = self.refactor_ops.to(device)
        return self


class SentenceLabelRowOperations():
    def __init__(self, labels_dict):
        super().__init__(labels_dict)
        sentence_operations_list = [
            labels_dict['is_deleted'],
            labels_dict['is_edited'],
            labels_dict['is_unchanged']
        ]
        sentence_operations_matrix = torch.tensor(sentence_operations_list, dtype=torch.long).T
        self.sentence_operations = torch.where(sentence_operations_matrix == 1)[1]
        self.deleted = (self.sentence_operations == 0).to(int)
        self.edited = (self.sentence_operations == 1).to(int)
        self.unchanged = (self.sentence_operations == 2).to(int)

    def to(self, device):
        super().to(device)
        self.sentence_operations = self.sentence_operations.to(device)
        self.deleted = self.deleted.to(device)
        self.edited = self.edited.to(device)
        self.unchanged = self.unchanged.to(device)
        return self


class SentenceLabelRowBlank():
    def __init__(self, *args, **kwargs):
        super().__init__()

    def to(self, device):
        return self

    # def __reduce__(self):
    #     return (self.__class__, (), dict(self.__dict__))


class S1(SentenceLabelRowAddition, SentenceLabelRowBlank):
    pass


class S12(SentenceLabelRowAddition, SentenceLabelRowRefactorRegression, SentenceLabelRowBlank):
    pass


class S124(SentenceLabelRowAddition, SentenceLabelRowRefactorRegression, SentenceLabelRowOperations, SentenceLabelRowBlank):
    pass


class S13(SentenceLabelRowAddition, SentenceLabelRowRefactorClassification, SentenceLabelRowBlank):
    pass

class S14(SentenceLabelRowAddition, SentenceLabelRowOperations, SentenceLabelRowBlank):
    pass



class S2(SentenceLabelRowRefactorRegression, SentenceLabelRowBlank):
    pass

class S3(SentenceLabelRowRefactorClassification, SentenceLabelRowBlank):
    pass

class S4(SentenceLabelRowOperations, SentenceLabelRowBlank):
    pass



def get_label_row_lame(config, labels_dict):
    if config.do_addition:
        return S1(labels_dict)
    if config.do_refactor:
        if config.do_regression:
            return S2(labels_dict)
        else:
            return S3(labels_dict)
    if config.do_operations:
        return S4(labels_dict)



def get_label_row(config, labels_dict):
    label_mixins = []
    if config.do_addition:
        label_mixins.append(SentenceLabelRowAddition)
    if config.do_refactor:
        if config.do_regression:
            label_mixins.append(SentenceLabelRowRefactorRegression)
        else:
            label_mixins.append(SentenceLabelRowRefactorClassification)
    if config.do_operations:
        label_mixins.append(SentenceLabelRowOperations)
    # blank
    label_mixins.append(SentenceLabelRowBlank)
    return Mixer(labels_dict=labels_dict, mixin=label_mixins)

