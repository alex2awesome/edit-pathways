from torch import nn
from torchmetrics import MeanSquaredError, F1
from modeling.utils_mixer import Mixer


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


class SentenceMetricsRefactorRegression(SentenceMetricsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = self.step + ': Refactor Distance, MSE'
        self.metrics[self.key] = MeanSquaredError(dist_sync_on_step=self.dist_sync_on_step)

    def update(self, pred, target):
        super().update(pred, target)
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
            self.step + ': Deletion F1',
            self.step + ': Edited F1',
            self.step + ': Unchanged F1'
        ]
        self.metrics[self.keys[0]] = F1(num_classes=3, average='weighted', dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[1]] = F1(num_classes=3, average='macro', dist_sync_on_step=self.dist_sync_on_step)
        # Binary Classification
        self.metrics[self.keys[2]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[3]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)
        self.metrics[self.keys[4]] = F1(num_classes=1, dist_sync_on_step=self.dist_sync_on_step)

    def update(self, pred, target):
        super().update(pred, target)
        self.metrics[self.keys[0]](pred.pred_sent_ops, target.sentence_operations)
        self.metrics[self.keys[1]](pred.pred_sent_ops, target.sentence_operations)
        self.metrics[self.keys[2]](pred.deleted, target.deleted)
        self.metrics[self.keys[3]](pred.edited, target.edited)
        self.metrics[self.keys[4]](pred.unchanged, target.unchanged)


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