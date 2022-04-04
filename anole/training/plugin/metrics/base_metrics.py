from typing import Union, Optional, TYPE_CHECKING
from typing_extensions import Protocol
from torch import Tensor

from ..base_plugin import BasePlugin

if TYPE_CHECKING:
    from ...strategy import BaseStrategy

__all__ = ['BaseMetric', 'MetricValue', 'MetricPlugin']

MetricType = Union[float, int, Tensor]


class BaseMetric(Protocol):
    """
    Definition of a standalone metric.
    A standalone metric exposes methods to reset its internal state and
    to emit a result. Emitting a result does not automatically cause
    a reset in the internal state.
    The specific metric implementation exposes ways to update the internal
    state. Usually, standalone metrics like :class:`Sum`, :class:`Mean`,
    :class:`Accuracy`, ... expose an `update` method.
    The `Metric` class can be used as a standalone metric by directly calling
    its methods.
    In order to automatically integrate the metric with the training and
    evaluation flows, you can use :class:`PluginMetric` class. The class
    receives events directly from the :class:`EvaluationPlugin` and can
    emits values on each callback. Usually, an instance of `Metric` is
    created within `PluginMetric`, which is then responsible for its
    update and results. See :class:`PluginMetric` for more details.
    """

    def result(self, **kwargs):
        """
        Obtains the value of the metric.
        :return: The value of the metric.
        """
        pass

    def reset(self, **kwargs):
        """
        Resets the metric internal state.
        :return: None.
        """
        pass


class MetricValue(object):

    def __init__(self, origin: 'BaseMetric', name: str, value: MetricType, x_plot: int):
        self.origin = origin
        self.name = name
        self.value = value
        self.x_plot = x_plot


class MetricPlugin(BasePlugin):
    """
    This class provides a generic implementation of a Plugin Metric.
    The user can subclass this class to easily implement custom plugin
    metrics.
    """

    def __init__(
        self,
        metric,
        reset_at: Optional[str] = 'epoch',
        emit_at: Optional[str] = 'epoch',
        mode: str = 'eval',
    ):
        super(MetricPlugin, self).__init__()
        assert mode in {'train', 'eval'}
        if mode == 'train':
            assert reset_at in {'iteration', 'epoch', 'whole'}
            assert emit_at in {'iteration', 'epoch', 'whole'}
        else:
            assert reset_at in {'iteration', 'instance'}
            assert emit_at in {'iteration', 'instance'}
        self._metric = metric
        self._mode = mode
        self._reset_at = reset_at
        self._emit_at = emit_at

    def reset(self, strategy=None):
        self._metric.reset()

    def result(self, strategy=None):
        return self._metric.result()

    def update(self, strategy=None):
        pass

    def _package_result(self, strategy: 'BaseStrategy'):
        metric_value = self.result(strategy)
        plot_x_position = strategy.clock.total_iterations
        metrics = []
        # DEBUG: print(metric_value)
        for k, v in metric_value.items():
            metric_name = str(k)
            metrics.append(MetricValue(self, metric_name, v, plot_x_position))
        return metrics

    # *************** Training Phase ***************
    def before_training(self, strategy: 'BaseStrategy'):
        super().before_training(strategy)
        if self._reset_at == 'whole' and self._mode == 'train':
            self.reset()

    def after_training(self, strategy: 'BaseStrategy'):
        super().after_training(strategy)
        if self._emit_at == 'whole' and self._mode == 'train':
            return self._package_result(strategy)

    def before_training_epoch(self, strategy: 'BaseStrategy'):
        super().before_training_epoch(strategy)
        if self._reset_at == 'epoch' and self._mode == 'train':
            self.reset()

    def after_training_epoch(self, strategy: 'BaseStrategy'):
        super().after_training_epoch(strategy)
        if self._emit_at == 'epoch' and self._mode == 'train':
            return self._package_result(strategy)

    def before_training_iteration(self, strategy: 'BaseStrategy'):
        super().before_training_iteration(strategy)
        if (self._reset_at == 'iteration' and self._mode == 'train'
                and strategy.clock.check_iteration(strategy, True)):
            self.reset()

    def after_training_iteration(self, strategy: 'BaseStrategy'):
        super().after_training_iteration(strategy)
        if self._mode == 'train':
            self.update(strategy)
        if (self._emit_at == 'iteration' and self._mode == 'train'
                and strategy.clock.check_iteration(strategy, False)):
            return self._package_result(strategy)

    # *************** Eval Phase ***************
    def before_eval(self, strategy: 'BaseStrategy'):
        super().before_eval(strategy)
        if self._reset_at == 'instance' and self._mode == 'eval':
            self.reset(strategy)

    def after_eval(self, strategy: 'BaseStrategy'):
        super().after_eval(strategy)
        if self._emit_at == 'instance' and self._mode == 'eval':
            return self._package_result(strategy)

    def before_eval_iteration(self, strategy: 'BaseStrategy'):
        super().before_eval_iteration(strategy)
        if self._reset_at == 'iteration' and self._mode == 'eval':
            self.reset(strategy)

    def after_eval_iteration(self, strategy: 'BaseStrategy'):
        super().after_eval_iteration(strategy)
        if self._mode == 'eval':
            self.update(strategy)
        if self._emit_at == 'iteration' and self._mode == 'eval':
            return self._package_result(strategy)
