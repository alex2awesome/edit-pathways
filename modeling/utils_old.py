import torch


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