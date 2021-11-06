from torch import nn
from torchmetrics import MeanSquaredError, F1, Precision, Recall
from modeling.utils_mixer import Mixer
import torch


class SentenceMetricsBase(nn.Module):
    def __init__(self, config, step, dist_sync_on_step=False):
        super().__init__()
        self.step = step
        self.dist_sync_on_step = dist_sync_on_step
        self.config = config
        self.metrics = nn.ModuleDict()

    def __call__(self, y_pred, y_true, *args, **kwargs):
        assert (isinstance(y_pred, list) and isinstance(y_true, list))
        for y_p_i, y_t_i in zip(y_pred, y_true):
            self.update(y_p_i, y_t_i)

    def reset(self):
        for k in self.metrics:
            self.metrics[k].reset()

    def compute(self):
        output = {}
        for k in self.metrics:
            output[k] = self.metrics[k].compute()
        return output

    def to_list(self, x):
        return x.detach().numpy().tolist()

class SentenceMetricsRefactorRegression(SentenceMetricsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = self.step + ': Refactor Distance, MSE'
        self.metrics[self.key] = MeanSquaredError(dist_sync_on_step=self.dist_sync_on_step)

    def update(self, pred, target):
        super().update(pred, target)
        print(str({
            'pred_refactored_ops': self.to_list(pred.pred_refactored),
            'label_refactored_ops': self.tolist(target.refactor_distance),
        }))
        self.metrics[self.key](pred.pred_refactored, target.refactor_distance)


class SentenceMetricsRefactorClassification(SentenceMetricsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = [
            self.step + ': Refactor Changes, Weighted',
            self.step + ': Refactor Changes, Macro',
            self.step + ': Refactor Up, F1',
            self.step + ': Refactor Unchanged, F1',
            self.step + ': Refactor Down, F1',
        ]
        self.metrics[self.keys[0]] = F1(num_classes=3, average='weighted', dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[1]] = F1(num_classes=3, average='macro', dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[2]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[3]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[4]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)

    def update(self, pred, target):
        super().update(pred, target)
        print(str({
            'pred_refactored_ops': self.to_list(pred.pred_refactored_ops),
            'label_refactored_ops': self.to_list(target.refactor_ops)
        }))
        self.metrics[self.keys[0]](pred.pred_refactored_ops, target.refactor_ops)
        self.metrics[self.keys[1]](pred.pred_refactored_ops, target.refactor_ops)
        self.metrics[self.keys[2]](pred.pred_ref_up, target.refactor_up)
        self.metrics[self.keys[3]](pred.pred_ref_un, target.refactor_unchanged)
        self.metrics[self.keys[4]](pred.pred_ref_down, target.refactor_down)


class SentenceMetricsOperations(SentenceMetricsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Multiclass Classification
        self.keys = [
            self.step + ': Sentence Changes, Weighted',
            self.step + ': Sentence Changes, Macro',
            self.step + ': Deletion F1 Full',
            self.step + ': Edited F1 Full',
            self.step + ': Unchanged F1 Full',
            #
            self.step + ': Deletion F1 +1',
            self.step + ': Edited F1 +1',
            self.step + ': Unchanged F1 +1',
            #
            self.step + ': Deletion F1 -1',
            self.step + ': Edited F1 -1',
            self.step + ': Unchanged -1',
            #
            self.step + ': Deletion F1 Prec',
            self.step + ': Edited F1 Prec',
            self.step + ': Unchanged F1 Prec',
            #
            self.step + ': Deletion F1 Recall',
            self.step + ': Edited F1 Recall',
            self.step + ': Unchanged F1 Recall'
        ]
        self.metrics[self.keys[0]] = F1(num_classes=3, average='weighted', dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[1]] = F1(num_classes=3, average='macro', dist_sync_on_step=self.dist_sync_on_step)
        # Binary Classification (F1 Full)
        self.metrics[self.keys[2]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[3]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[4]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        # Binary Classification (F1 +1)
        self.metrics[self.keys[5]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[6]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[7]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        # Binary Classification (F1 -1)
        self.metrics[self.keys[8]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[9]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[10]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        # Binary Classification (Precision)
        self.metrics[self.keys[11]] = Precision(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[12]] = Precision(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[13]] = Precision(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        # Binary Classification (Recall)
        self.metrics[self.keys[14]] = Recall(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[15]] = Recall(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[16]] = Recall(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)

    def update(self, pred, target):
        super().update(pred, target)
        print(str({
            'pred_sentence_ops': self.to_list(pred.pred_sent_ops),
            'label_sentence_ops': self.to_list(target.sentence_operations)
        }))
        self.metrics[self.keys[0]](pred.pred_sent_ops, target.sentence_operations)
        self.metrics[self.keys[1]](pred.pred_sent_ops, target.sentence_operations)
        # Binary Classification F1 Full
        self.metrics[self.keys[2]](pred.deleted, target.deleted)
        self.metrics[self.keys[3]](pred.edited, target.edited)
        self.metrics[self.keys[4]](pred.unchanged, target.unchanged)
        # F1 +1
        del_1 = torch.where(target.deleted == 1)
        self.metrics[self.keys[5]](pred.deleted[del_1], target.deleted[del_1])
        edit_1 = torch.where(target.edited == 1)
        self.metrics[self.keys[6]](pred.edited[edit_1], target.edited[edit_1])
        unch_1 = torch.where(target.unchanged == 1)
        self.metrics[self.keys[7]](pred.unchanged[unch_1], target.unchanged[unch_1])
        # F1 -1
        del_0 = torch.where(target.deleted == 0)
        self.metrics[self.keys[8]](pred.deleted[del_0], target.deleted[del_0])
        edit_0 = torch.where(target.edited == 0)
        self.metrics[self.keys[9]](pred.edited[edit_0], target.edited[edit_0])
        unch_0 = torch.where(target.unchanged == 0)
        self.metrics[self.keys[10]](pred.unchanged[unch_0], target.unchanged[unch_0])
        # Precision
        self.metrics[self.keys[11]](pred.deleted, target.deleted)
        self.metrics[self.keys[12]](pred.edited, target.edited)
        self.metrics[self.keys[13]](pred.unchanged, target.unchanged)
        # Recall
        self.metrics[self.keys[14]](pred.deleted, target.deleted)
        self.metrics[self.keys[15]](pred.edited, target.edited)
        self.metrics[self.keys[16]](pred.unchanged, target.unchanged)


class SentenceMetricsAddRegression(SentenceMetricsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = [
            self.step + ': Additions Above, MSE',
            self.step + ': Additions Below, MSE',
        ]
        self.metrics[self.keys[0]] = MeanSquaredError(dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[1]] = MeanSquaredError(dist_sync_on_step=self.dist_sync_on_step)

    def update(self, pred, target):
        super().update(pred, target)
        print(str({
            'pred_added_before': self.to_list(pred.pred_added_before),
            'label_added_before': self.to_list(target.num_add_before)
        }))
        print(str({
            'pred_added_after': self.to_list(pred.pred_added_after),
            'label_added_after': self.to_list(target.num_add_after)
        }))
        self.metrics[self.keys[0]](pred.pred_added_before, target.num_add_before)
        self.metrics[self.keys[1]](pred.pred_added_after, target.num_add_after)


class SentenceMetricsAddClassification(SentenceMetricsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = [
            self.step + ': Additions Above, F1',
            self.step + ': Additions Below, F1',
        ]
        self.metrics[self.keys[0]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[1]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)

    def update(self, pred, target):
        super().update(pred, target)
        print(str({
            'pred_added_before': self.to_list(pred.pred_added_before),
            'label_added_before': self.to_list(target.num_add_before),
        }))
        print(str({
            'pred_added_after': self.to_list(pred.pred_added_after),
            'label_added_after': self.to_list(target.num_add_after)
        }))
        self.metrics[self.keys[0]](pred.pred_added_before, target.num_add_before)
        self.metrics[self.keys[1]](pred.pred_added_after, target.num_add_after)


class SentenceMetricsBlank(SentenceMetricsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, pred, target):
        pass




def get_sentence_metrics(config, step, dist_sync_on_step):
    sentence_metrics_mixin = []
    if config.do_addition:
        if config.do_regression:
            sentence_metrics_mixin.append(SentenceMetricsAddRegression)
        else:
            sentence_metrics_mixin.append(SentenceMetricsAddClassification)
    if config.do_refactor:
        if config.do_regression:
            sentence_metrics_mixin.append(SentenceMetricsRefactorRegression)
        else:
            sentence_metrics_mixin.append(SentenceMetricsRefactorClassification)
    if config.do_operations:
        sentence_metrics_mixin.append(SentenceMetricsOperations)
    sentence_metrics_mixin.append(SentenceMetricsBlank)
    return Mixer(
        config=config,
        step=step,
        dist_sync_on_step=dist_sync_on_step,
        mixin=sentence_metrics_mixin
    )