import warnings
from copy import copy
from collections import defaultdict
from typing import Union, Sequence, TYPE_CHECKING

from .metrics import loss_metrics
from .base_plugin import BasePlugin
if TYPE_CHECKING:
    from .metrics import MetricPlugin
    from ..strategy import BaseStrategy
from .logger import TextLogger, BaseLogger

__all__ = ['EvaluationPlugin', 'default_logger']


class EvaluationPlugin(BasePlugin):
    """
    An evaluation plugin that obtains relevant data from the
    training and eval loops of the strategy through callbacks.
    The plugin keeps a dictionary with the last recorded value for each metric.
    The dictionary will be returned by the `train` and `eval` methods of the
    strategies.
    It is also possible to keep a dictionary with all recorded metrics by
    specifying `collect_all=True`. The dictionary can be retrieved via
    the `get_all_metrics` method.
    This plugin also logs metrics using the provided loggers.
    """
    def __init__(
        self,
        *metrics: Union['MetricPlugin', Sequence['MetricPlugin']],
        loggers: Union['BaseLogger', Sequence['BaseLogger']] = None,
    ):
        """
        Creates an instance of the evaluation plugin.
        :param metrics: The metrics to compute.
        :param loggers: The loggers to be used to log the metric values.
        :param collect_all: if True, collect in a separate dictionary all
            metric curves values. This dictionary is accessible with
            `get_all_metrics` method.
        :param benchmark: continual learning benchmark needed to check stream
            completeness during evaluation or other kind of properties. If
            None, no check will be conducted and the plugin will emit a
            warning to signal this fact.
        :param strict_checks: if True, `benchmark` has to be provided.
            In this case, only full evaluation streams are admitted when
            calling `eval`. An error will be raised otherwise. When False,
            `benchmark` can be `None` and only warnings will be raised.
        :param suppress_warnings: if True, warnings and errors will never be
            raised from the plugin.
            If False, warnings and errors will be raised following
            `benchmark` and `strict_checks` behavior.
        """
        super().__init__()
        self.metrics = metrics
        if loggers is None:
            loggers = []
        elif not isinstance(loggers, Sequence):
            loggers = [loggers]
        self.loggers: Sequence['BaseLogger'] = loggers

        if len(self.loggers) == 0:
            warnings.warn('No loggers specified, metrics will not be logged')

        self.all_metric_results = defaultdict(lambda: ([], []))
        self.last_metric_results = {}

    def _update_metrics(self, strategy: 'BaseStrategy', callback: str):
        # Execute the related updated function
        metric_values = []
        for metric in self.metrics:
            for submetric in metric:
                metric_result = getattr(submetric, callback)(strategy)
                if metric_result is not None:
                    for m in metric_result:
                        metric_values.append(m)

        for metric_value in metric_values:
            name = metric_value.name
            x = metric_value.x_plot
            val = metric_value.value
            self.all_metric_results[name][0].append(x)
            self.all_metric_results[name][1].append(val)
            self.last_metric_results[name] = val

        for logger in self.loggers:
            getattr(logger, callback)(strategy, metric_values)

    def get_last_metrics(self):
        """
        Return a shallow copy of dictionary with metric names
        as keys and last metrics value as values.
        :return: a dictionary with full metric
            names as keys and last metric value as value.
        """
        return copy(self.last_metric_results)

    def get_all_metrics(self):
        return self.all_metric_results

    def reset_last_metrics(self):
        """
        Set the dictionary storing last value for each metric to be
        empty dict.
        """
        self.last_metric_results = {}

    # *************** Training Phase ***************
    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_training')

    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_training')

    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_training_epoch')

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_training_epoch')

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_training_iteration')

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_training_iteration')

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_forward')

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_forward')

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_backward')

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_backward')

    # *************** Eval Phase ***************
    def before_eval(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_eval')

    def after_eval(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_eval')

    def before_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'before_eval_iteration')

    def after_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self._update_metrics(strategy, 'after_eval_iteration')


default_logger = EvaluationPlugin(
    loss_metrics(iteration=True, epoch=True, whole=True, eval=True),
    loggers=[TextLogger()],
)
