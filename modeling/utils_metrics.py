from torch import nn
from torchmetrics import F1, MeanSquaredError


class SentenceMetrics(nn.Module):
    def __init__(self, config, step, device, dist_sync_on_step):
        super().__init__()
        self.step = step
        self.config = config

        # Multiclass Classification
        self.sentence_changes_weighted = F1(num_classes=3, average='weighted', dist_sync_on_step=dist_sync_on_step)
        self.sentence_changes_macro = F1(num_classes=3, average='macro', dist_sync_on_step=dist_sync_on_step)
        # Binary Classification
        self.deletion = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)
        self.edited = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)
        self.unchanged = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)
        # Regression
        if self.config.do_regression:
            self.additions_below = MeanSquaredError(dist_sync_on_step=dist_sync_on_step)
            self.additions_above = MeanSquaredError(dist_sync_on_step=dist_sync_on_step)
            self.refactor_distance = MeanSquaredError(dist_sync_on_step=dist_sync_on_step)
        else:
            self.additions_below = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)
            self.additions_above = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)
            self.refactor_ops_weighted = F1(num_classes=3, average='weighted', dist_sync_on_step=dist_sync_on_step)
            self.refactor_ops_macro = F1(num_classes=3, average='macro', dist_sync_on_step=dist_sync_on_step)
            self.refactor_up = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)
            self.refactor_un = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)
            self.refactor_down = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)

    def to(self, device):
        self.sentence_changes_weighted = self.sentence_changes_weighted.to(device)
        self.sentence_changes_macro = self.sentence_changes_macro.to(device)
        self.deletion = self.deletion.to(device)
        self.edited = self.edited.to(device)
        self.unchanged = self.unchanged.to(device)
        self.additions_below = self.additions_below.to(device)
        self.additions_above = self.additions_above.to(device)
        if self.config.do_regression:
            self.refactor_distance = self.refactor_distance.to(device)
        else:
            self.refactor_ops_weighted = self.refactor_ops_weighted.to(device)
            self.refactor_ops_macro = self.refactor_ops_macro.to(device)
            self.refactor_up = self.refactor_up.to(device)
            self.refactor_un = self.refactor_un.to(device)
            self.refactor_down = self.refactor_down.to(device)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        assert (
                (isinstance(y_true, list) and isinstance(y_pred, list)) or
                (not isinstance(y_true, list) and not isinstance(y_pred, list))
        )
        if isinstance(y_true, list) and isinstance(y_pred, list):
            # if we don't organize PredRows/LabelRows as a Batch (maybe has memory issues?)
            for y_t_i, y_p_i in zip(y_true, y_pred):
                self.sentence_changes_weighted(y_p_i.pred_sent_ops, y_t_i.sentence_operations)
                self.sentence_changes_macro(y_p_i.pred_sent_ops, y_t_i.sentence_operations)
                self.deletion(y_p_i.deleted, y_t_i.deleted)
                self.edited(y_p_i.edited, y_t_i.edited)
                self.unchanged(y_p_i.unchanged, y_t_i.unchanged)
                self.additions_above(y_p_i.pred_added_before, y_t_i.num_add_before)
                self.additions_below(y_p_i.pred_added_after, y_t_i.num_add_after)
                if self.config.do_regression:
                    self.refactor_distance(y_p_i.pred_refactored, y_t_i.refactor_distance)
                else:
                    self.refactor_ops_weighted(y_p_i.pred_refactored_ops, y_t_i.refactor_ops)
                    self.refactor_ops_macro(y_p_i.pred_refactored_ops, y_t_i.refactor_ops)
                    self.refactor_up(y_p_i.pred_ref_up, y_t_i.refactor_up)
                    self.refactor_un(y_p_i.pred_ref_un, y_t_i.refactor_unchanged)
                    self.refactor_down(y_p_i.pred_ref_down, y_t_i.refactor_down)

        # if we do organize PredRows/LabelRows as a Batch
        elif (not isinstance(y_pred, list)) and (not isinstance(y_true, list)):
            self.sentence_changes_weighted(y_pred.sent_ops, y_true.sentence_operations)
            self.sentence_changes_macro(y_pred.sent_ops, y_true.sentence_operations)
            self.deletion(y_pred.deleted, y_true.deleted)
            self.edited(y_pred.edited, y_true.edited)
            self.unchanged(y_pred.unchanged, y_true.unchanged)
            self.additions_above(y_pred.add_before, y_true.num_add_before)
            self.additions_below(y_pred.add_after, y_true.num_add_after)
            self.refactor_distance(y_pred.refactored, y_true.refactor_distance)

    def compute(self):
        output = {}
        output['%s: Sentence Changes, Weighted' % self.step] = self.sentence_changes_weighted.compute()
        output['%s: Sentence Changes, Macro' % self.step] = self.sentence_changes_macro.compute()
        output['%s: Deletion F1' % self.step] = self.deletion.compute()
        output['%s: Edited F1' % self.step] = self.edited.compute()
        output['%s: Unchanged F1' % self.step] = self.unchanged.compute()
        output['%s: Additions Above, MSE' % self.step] = self.additions_above.compute()
        output['%s: Additions Below, MSE' % self.step] = self.additions_below.compute()
        if self.config.do_regression:
            output['%s: Refactor Distance, MSE' % self.step] = self.refactor_distance.compute()
        else:
            output['%s: Refactor Changes, Weighted' % self.step] = self.refactor_ops_weighted.compute()
            output['%s: Refactor Changes, Macro' % self.step] = self.refactor_ops_macro.compute()
            output['%s: Refactor Up, F1' % self.step] = self.refactor_up.compute()
            output['%s: Refactor Unchanged, F1' % self.step] = self.refactor_un.compute()
            output['%s: Refactor Down, F1' % self.step] = self.refactor_down.compute()
        return output

    def reset(self):
        self.sentence_changes_weighted.reset()
        self.sentence_changes_macro.reset()
        self.deletion.reset()
        self.edited.reset()
        self.unchanged.reset()
        self.additions_above.reset()
        self.additions_below.reset()
        if self.config.do_regression:
            self.refactor_distance.reset()
        else:
            self.refactor_ops_weighted.reset()
            self.refactor_ops_macro.reset()
            self.refactor_up.reset()
            self.refactor_un.reset()
            self.refactor_down.reset()


class DocMetrics(nn.Module):
    def __init__(self, step, config, dist_sync_on_step):
        super().__init__()
        self.step = step
        self.metrics = {}
        if config.do_regression:
            for k in config.id2label:
                self.metrics[k] = MeanSquaredError(dist_sync_on_step=dist_sync_on_step)
        else:
            for k in config.id2label:
                self.metrics[k] = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        assert (isinstance(y_true, list) and isinstance(y_pred, list))
        for y_p_i, y_t_i in zip(y_pred, y_true):
            for y_p_i_j, y_t_i_j in zip(y_p_i.preds, y_t_i.labels):
                for k in self.metrics:
                    self.metrics[k](y_p_i_j.unsqueeze(dim=0), y_t_i_j.unsqueeze(dim=0))

    def compute(self):
        output = {}
        for k in self.metrics:
            output_key = '%s: %s' % (self.step, k)
            output[output_key] = self.metrics[k].compute()
        return output

    def reset(self):
        for k in self.metrics:
            self.metrics[k].reset()

    def to(self, device):
        for k in self.metrics:
            self.metrics[k].to(device)
