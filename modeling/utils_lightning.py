import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection, F1, MeanSquaredError
from modeling.utils_general import get_config, SuperBlank
from torch import nn
adam_beta1, adam_beta2, adam_epsilon = .9, .999, 1e-08
class LightningOptimizer(SuperBlank, pl.LightningModule):
    """
    Contains logic for optimization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)

        #
        self.lr = self.config.learning_rate
        self.num_warmup_steps = self.config.num_warmup_steps
        self.dataset_size = self.config.num_steps_per_epoch

    # optimization
    def _lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.num_warmup_steps))
        return 1.0

    def _lr_lambda_linear(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        num = self.num_training_steps - current_step
        denom = self.num_training_steps - self.num_warmup_steps
        num = float(max(0, num))
        denom = float(max(1, denom))
        return num / denom

    def configure_optimizers(self):
        self.num_training_steps = self.dataset_size * self.trainer.max_epochs
        optimizer_kwargs = {
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,
        }
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **optimizer_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda_linear),
                'interval': 'step',
            }
        }


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
            self.refactor_distance = F1(num_classes=1, dist_sync_on_step=dist_sync_on_step)

        print(device)
        # self.to(device)

    def to(self, device):
        self.sentence_changes_weighted = self.sentence_changes_weighted.to(device)
        self.sentence_changes_macro = self.sentence_changes_macro.to(device)
        self.deletion = self.deletion.to(device)
        self.edited = self.edited.to(device)
        self.unchanged = self.unchanged.to(device)
        self.additions_below = self.additions_below.to(device)
        self.additions_above = self.additions_above.to(device)
        self.refactor_distance = self.refactor_distance.to(device)

    def __call__(self, y_pred, y_true, *args, **kwargs):
        self.sentence_changes_weighted(y_pred.sent_ops, y_true.sentence_operations)
        self.sentence_changes_macro(y_pred.sent_ops, y_true.sentence_operations)
        self.deletion(y_pred.deleted, y_true.deleted)
        self.edited(y_pred.edited, y_true.edited)
        self.unchanged(y_pred.unchanged, y_true.unchanged)
        self.additions_above(y_pred.add_before, y_true.num_add_before)
        self.additions_below(y_pred.add_after, y_true.num_add_after)
        self.refactor_distance(y_pred.refactored, y_true.refactor_distance)

    def compute(self):
        return {
            '%s: Sentence Changes, Weighted' % self.step : self.sentence_changes_weighted.compute(),
            '%s: Sentence Changes, Macro' % self.step: self.sentence_changes_macro.compute(),
            '%s: Deletion F1' % self.step: self.deletion.compute(),
            '%s: Edited F1' % self.step: self.edited.compute(),
            '%s: Unchanged F1' % self.step: self.unchanged.compute(),
            '%s: Additions Above, MSE' % self.step: self.additions_above.compute(),
            '%s: Additions Below, MSE' % self.step: self.additions_below.compute(),
            '%s: Refactor Distance, MSE' % self.step: self.refactor_distance.compute(),
        }

    def reset(self):
        self.sentence_changes_weighted.reset()
        self.sentence_changes_macro.reset()
        self.deletion.reset()
        self.edited.reset()
        self.unchanged.reset()
        self.additions_above.reset()
        self.additions_below.reset()
        self.refactor_distance.reset()


class DocMetrics():
    def __init__(self, dist_sync_on_step):
        self.additions = MeanSquaredError(dist_sync_on_step=dist_sync_on_step)
        self.deletions = MeanSquaredError(dist_sync_on_step=dist_sync_on_step)
        self.edited = MeanSquaredError(dist_sync_on_step=dist_sync_on_step)
        self.refactored = MeanSquaredError(dist_sync_on_step=dist_sync_on_step)
        self.unchanged = MeanSquaredError(dist_sync_on_step=dist_sync_on_step)


class LightningStepsBase(SuperBlank, pl.LightningModule):
    """Mixin to handle Lightning hooks and metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)

        #####
        # metrics
        self.task_type = self.config.experiment
        if self.task_type == 'document':
            self.training_report = DocMetrics(
                dist_sync_on_step=kwargs.get('accelerator') == 'dp'
            )
            self.validation_report = DocMetrics(
                dist_sync_on_step=kwargs.get('accelerator') == 'dp'
            )
        if self.task_type == 'sentence':
            self.training_report = SentenceMetrics(
                config=self.config,
                step='Train',
                device=self.device,
                dist_sync_on_step=kwargs.get('accelerator') == 'dp'
            )
            self.validation_report = SentenceMetrics(
                config=self.config,
                step='Validation',
                device=self.device,
                dist_sync_on_step=kwargs.get('accelerator') == 'dp'
            )

    def step(self, batch, batch_idx, log_name):
        loss, y_pred = self.forward(**batch)
        self.log(log_name, loss)
        return {'loss': loss, 'y_pred': y_pred, 'y_true': batch['labels']}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'Training Loss')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'Validation loss')

    def training_step_end(self, batch_parts):
        # if run on multi-GPUs, this is a list(?)
        if isinstance(batch_parts, list):
            for batch in batch_parts:
                self.training_report(batch['y_pred'], batch['y_true'])
            return sum(map(lambda x: x['loss'], batch_parts))
        else:
            self.training_report(batch_parts['y_pred'], batch_parts['y_true'])
            return batch_parts['loss']

    def validation_step_end(self, batch_parts):
        if isinstance(batch_parts, list):
            for batch in batch_parts:
                self.validation_report(batch['y_pred'], batch['y_true'])
        else:
            self.validation_report(batch_parts['y_pred'], batch_parts['y_true'])

    def training_epoch_end(self, outputs):
        report = self.training_report.compute()
        self.log_dict(report)
        self.training_report.reset()

    def validation_epoch_end(self, outputs):
        report = self.validation_report.compute()
        self.log_dict(report)# report['Sentence Changes, Weighted'])
        # self.log_dict('Validation Report', report)  # report['Sentence Changes, Weighted'])
        self.validation_report.reset()

    # def on_validation_end(self):
    #     try:
    #         self.log('hp_metric', max(self.hp_metric_list))
    #     except MisconfigurationException:
    #         pass
