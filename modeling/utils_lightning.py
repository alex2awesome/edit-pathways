import pytorch_lightning as pl
import torch
from modeling.utils_general import get_config, SuperBlank

from modeling.utils_metrics import SentenceMetrics, DocMetrics

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


class LightningStepsBase(SuperBlank, pl.LightningModule):
    """Mixin to handle Lightning hooks and metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, batch_idx, log_name):
        loss, y_pred = self.forward(**batch)
        self.log(log_name, loss)
        torch.cuda.empty_cache()
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
        self.log_dict(report)
        self.validation_report.reset()


class LightningStepsDoc(LightningStepsBase):
    """Mixin to handle Lightning hooks and metrics"""

    def __init__(self, *args, **kwargs):
        self.config = get_config(kwargs=kwargs)
        super().__init__(*args, **kwargs)

        #####
        # metrics
        dist_sync_on_step = kwargs.get('accelerator') == 'dp'
        self.task_type = self.config.experiment
        self.training_report = DocMetrics(dist_sync_on_step=dist_sync_on_step, step='Train', config=self.config)
        self.validation_report = DocMetrics(dist_sync_on_step=dist_sync_on_step, step='Validation', config=self.config)


class LightningStepsSentence(LightningStepsBase):
    """Mixin to handle Lightning hooks and metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = get_config(kwargs=kwargs)

        #####
        # metrics
        dist_sync_on_step = kwargs.get('accelerator') == 'dp'
        self.task_type = self.config.experiment
        self.training_report = SentenceMetrics(
            config=self.config,
            step='Train',
            device=self.device,
            dist_sync_on_step=dist_sync_on_step
        )
        self.validation_report = SentenceMetrics(
            config=self.config,
            step='Validation',
            device=self.device,
            dist_sync_on_step=dist_sync_on_step
        )