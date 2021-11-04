from torch import nn
from torchmetrics import F1, MeanSquaredError


class DocMetrics(nn.Module):
    def __init__(self, step, config, dist_sync_on_step):
        super().__init__()
        self.step = step
        self.metrics = nn.ModuleDict()
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
