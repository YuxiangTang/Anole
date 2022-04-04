from typing import Dict
from collections import defaultdict, ChainMap

import torch
from torch import Tensor

from .base_metrics import BaseMetric, MetricPlugin
from .counter import Min, Max
from ...builder import METRICSPLUGIN
from anole.utils import error_evaluation

__all__ = ['BestRecorder', 'BestRecorderPluginMetric', 'best_recorder_metrics']


class BestRecorder(BaseMetric):
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

    def __init__(self, sort_="min"):
        """
        Creates an instance of the loss metric.
        By default this metric in its initial state will return a loss
        value of 0. The metric can be updated by using the `update` method
        while the running loss can be retrieved using the `result` method.
        """
        self.sort_ = sort_
        if self.sort_ == "min":
            self._recorder = defaultdict(Min)
        else:
            self._recorder = defaultdict(Max)
        self._recorder_list = defaultdict(list)
        """
        The mean utility that will be used to store the running accuracy
        for each task label.
        """

    @torch.no_grad()
    def update(self, loss: Tensor, name: int) -> None:
        """
        Update the running loss given the loss Tensor and the minibatch size.
        :param loss: The loss Tensor. Different reduction types don't affect
            the result.
        :param patterns: The number of patterns in the minibatch.
        :param dataset_label: the task label associated to the current experience
        :return: None.
        """
        self._recorder_list[name].append(loss.clone())

    def result(self, strategy=None, name=None) -> Dict[int, float]:
        """
        Retrieves the running average loss per pattern.
        Calling this method will not change the internal state of the metric.
        :param task_label: None to return metric values for all the task labels.
            If an int, return value only for that task label
        :return: The running loss, as a float.
        """
        assert (name is None or isinstance(name, str))

        def update_recorder(k, record_list):
            ret_dict = {}
            ret = error_evaluation(record_list)
            for sub_k, sub_val in ret.items():
                ret_dict[f"{k}_{sub_k}"] = sub_val
                flag = self._recorder[f"{k}_best"].update(ret_dict[f"{k}_Mean"])
                ret_dict[f"{k}_best"] = self._recorder[f"{k}_best"].result()
                if flag:
                    strategy.saver.save_ckpt(strategy, mode='best')
            return ret_dict

        if name is None:
            ret_dict = dict(ChainMap(*[update_recorder(k, v) for k, v in self._recorder_list.items()]))
        else:
            ret_dict = update_recorder(name, self._recorder_list[name])
        return ret_dict

    def reset(self, name=None) -> None:
        pass


class BestRecorderPluginMetric(MetricPlugin):

    def __init__(self, reset_at='instance', emit_at='instance', mode='eval', sort_='min'):
        self._recorder = BestRecorder(sort_)
        super(BestRecorderPluginMetric, self).__init__(
            self._recorder,
            reset_at,
            emit_at,
            mode,
        )

    def reset(self, strategy=None) -> None:
        if self._reset_at == 'instance' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(strategy.dataset.dataset_name)

    def result(self, strategy=None) -> float:
        if self._emit_at == 'instance' or strategy is None:
            return self._metric.result(strategy)
        else:
            return self._metric.result(strategy, strategy.dataset.dataset_name)

    def update(self, strategy):
        name = strategy.dataset.dataset_name
        self._recorder.update(strategy.loss, name=name)

    def __str__(self):
        return "Best_Recorder"


@METRICSPLUGIN.register_obj
def best_recorder_metrics(sort_='min') -> MetricPlugin:
    return [BestRecorderPluginMetric(sort_=sort_)]
