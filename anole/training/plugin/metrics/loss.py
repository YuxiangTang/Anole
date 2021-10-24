from typing import List, Dict
from collections import defaultdict

import torch
from torch import Tensor

from .base_metrics import BaseMetric, MetricPlugin
from .counter import Mean
from ...builder import METRICSPLUGIN

__all__ = [
    'Loss', 'LossPluginMetric', 'IterationLoss', 'EpochLoss', 'WholeLoss',
    'EvalLoss', 'loss_metrics'
]


class Loss(BaseMetric):
    """
    The standalone Loss metric. This is a general metric
    used to compute more specific ones.
    Instances of this metric keeps the running average loss
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.
    Each time `result` is called, this metric emits the average loss
    across all predictions made since the last `reset`.
    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return a loss value of 0.
    """
    def __init__(self):
        """
        Creates an instance of the loss metric.
        By default this metric in its initial state will return a loss
        value of 0. The metric can be updated by using the `update` method
        while the running loss can be retrieved using the `result` method.
        """
        self._mean_loss = defaultdict(Mean)
        """
        The mean utility that will be used to store the running accuracy
        for each task label.
        """

    @torch.no_grad()
    def update(self, loss: Tensor, patterns: int, name: int) -> None:
        """
        Update the running loss given the loss Tensor and the minibatch size.
        :param loss: The loss Tensor. Different reduction types don't affect
            the result.
        :param patterns: The number of patterns in the minibatch.
        :param dataset_label: the task label associated to the current experience
        :return: None.
        """
        self._mean_loss[name].update(torch.mean(loss.clone()), weight=patterns)

    def result(self, name=None) -> Dict[int, float]:
        """
        Retrieves the running average loss per pattern.
        Calling this method will not change the internal state of the metric.
        :param task_label: None to return metric values for all the task labels.
            If an int, return value only for that task label
        :return: The running loss, as a float.
        """
        assert (name is None or isinstance(name, str))
        if name is None:
            return {k: v.result() for k, v in self._mean_loss.items()}
        else:
            return {name: self._mean_loss[name].result()}

    def reset(self, name=None) -> None:
        """
        Resets the metric.
        :param task_label: None to reset all metric values. If an int,
            reset metric value corresponding to that task label.
        :return: None.
        """
        assert (name is None or isinstance(name, str))
        if name is None:
            self._mean_loss = defaultdict(Mean)
        else:
            self._mean_loss[name].reset()


class LossPluginMetric(MetricPlugin):
    def __init__(self, reset_at, emit_at, mode):
        self._loss = Loss()
        super(LossPluginMetric, self).__init__(self._loss, reset_at, emit_at,
                                               mode)

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'whole' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(strategy.dataset.dataset_name)

    def result(self, strategy=None) -> float:
        if self._emit_at == 'whole' or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(strategy.dataset.dataset_name)

    def update(self, strategy):
        name = strategy.dataset.dataset_name
        self._loss.update(strategy.loss, patterns=strategy.img.shape[0], name=name)


class IterationLoss(LossPluginMetric):
    def __init__(self):
        super(IterationLoss, self).__init__(reset_at='iteration',
                                            emit_at='iteration',
                                            mode='train')

    def __str__(self):
        return "Loss_MB"


class EpochLoss(LossPluginMetric):
    def __init__(self):
        super(EpochLoss, self).__init__(reset_at='epoch',
                                        emit_at='epoch',
                                        mode='train')

    def __str__(self):
        return "Loss_Ep"


class WholeLoss(LossPluginMetric):
    """
    At the end of the entire stream of experiences, this metric reports the
    average loss over all patterns seen in all experiences.
    This plugin metric only works at eval time.
    """
    def __init__(self):
        """
        Creates an instance of StreamLoss metric
        """
        super(WholeLoss, self).__init__(reset_at='whole',
                                        emit_at='whole',
                                        mode='train')

    def __str__(self):
        return "Loss_Whole"


class EvalLoss(LossPluginMetric):
    """
    At the end of the entire stream of experiences, this metric reports the
    average loss over all patterns seen in all experiences.
    This plugin metric only works at eval time.
    """
    def __init__(self):
        """
        Creates an instance of StreamLoss metric
        """
        super(EvalLoss, self).__init__(reset_at='instance',
                                       emit_at='instance',
                                       mode='eval')

    def __str__(self):
        return "Loss_Whole"


@METRICSPLUGIN.register_obj
def loss_metrics(*,
                 iteration=False,
                 epoch=False,
                 whole=False,
                 eval=False) -> List[MetricPlugin]:
    metrics = []
    if iteration:
        metrics.append(IterationLoss())

    if epoch:
        metrics.append(EpochLoss())

    if whole:
        metrics.append(WholeLoss())

    if eval:
        metrics.append(EvalLoss())

    return metrics
