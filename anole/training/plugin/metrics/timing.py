from abc import abstractmethod
import time
from typing import TYPE_CHECKING

from .base_metrics import BaseMetric, MetricPlugin
from ...builder import METRICSPLUGIN

if TYPE_CHECKING:
    from anole.training import BaseStrategy

__all__ = [
    'ElapsedTime', 'IterationTime', 'EpochTime', 'WholeTime', 'EvalTime',
    'timing_metrics'
]


class ElapsedTime(BaseMetric):
    """
    The standalone Elapsed Time metric.
    Instances of this metric keep track of the time elapsed between calls to the
    `update` method. The starting time is set when the `update` method is called
    for the first time. That is, the starting time is *not* taken at the time
    the constructor is invoked.
    Calling the `update` method more than twice will update the metric to the
    elapsed time between the first and the last call to `update`.
    The result, obtained using the `result` method, is the time, in seconds,
    computed as stated above.
    The `reset` method will set the metric to its initial state, thus resetting
    the initial time. This metric in its initial state (or if the `update`
    method was invoked only once) will return an elapsed time of 0.
    """
    def __init__(self):
        """
        Creates an instance of the ElapsedTime metric.
        This metric in its initial state (or if the `update` method was invoked
        only once) will return an elapsed time of 0. The metric can be updated
        by using the `update` method while the running accuracy can be retrieved
        using the `result` method.
        """
        self._init_time = None
        self._prev_time = None

    def update(self) -> None:
        """
        Update the elapsed time.
        For more info on how to set the initial time see the class description.
        :return: None.
        """
        now = time.perf_counter()
        if self._init_time is None:
            self._init_time = now
        self._prev_time = now

    def result(self) -> float:
        """
        Retrieves the elapsed time.
        Calling this method will not change the internal state of the metric.
        :return: The elapsed time, in seconds, as a float value.
        """
        if self._init_time is None:
            return 0.0
        return self._prev_time - self._init_time

    def reset(self) -> None:
        """
        Resets the metric, including the initial time.
        :return: None.
        """
        self._prev_time = None
        self._init_time = None


class TimePluginMetric(MetricPlugin):
    def __init__(self, reset_at, emit_at, mode):
        self._time = ElapsedTime()

        super(TimePluginMetric, self).__init__(self._time, reset_at, emit_at,
                                               mode)

    def update(self, strategy):
        self._time.update()

    def result(self, strategy=None):
        super().result(strategy=strategy)
        cost = self._time.result()
        return {str(self) + '_cost': cost}

    @abstractmethod
    def __str__(self):
        pass


class IterationTime(TimePluginMetric):
    """
    The minibatch time metric.
    This plugin metric only works at training time.
    This metric "logs" the elapsed time for each iteration.
    If a more coarse-grained logging is needed, consider using
    :class:`EpochTime`.
    """
    def __init__(self):
        """
        Creates an instance of the minibatch time metric.
        """
        super(IterationTime, self).__init__(reset_at='iteration',
                                            emit_at='iteration',
                                            mode='train')

    def before_training_iteration(self, strategy):
        super().before_training_iteration(strategy)
        self._time.update()

    def __str__(self):
        return "Time_Iteration"


class EpochTime(TimePluginMetric):
    """
    The epoch elapsed time metric.
    This plugin metric only works at training time.
    The elapsed time will be logged after each epoch.
    """
    def __init__(self):
        """
        Creates an instance of the epoch time metric.
        """

        super(EpochTime, self).__init__(reset_at='epoch',
                                        emit_at='epoch',
                                        mode='train')

    def before_training_epoch(self, strategy):
        super().before_training_epoch(strategy)
        self._time.update()

    def __str__(self):
        return "Time_Epoch"


class WholeTime(TimePluginMetric):
    """
    The experience time metric.
    This plugin metric only works at eval time.
    After each experience, this metric emits the average time of that
    experience.
    """
    def __init__(self):
        """
        Creates an instance of the experience time metric.
        """
        super(WholeTime, self).__init__(reset_at='whole',
                                        emit_at='whole',
                                        mode='train')

    def before_training(self, strategy: 'BaseStrategy'):
        super().before_training(strategy)
        self._time.update()

    def __str__(self):
        return "Time_Whole"


class EvalTime(TimePluginMetric):
    """
    The stream time metric.
    This metric only works at eval time.
    After the entire evaluation stream,
    this plugin metric emits the average time of that stream.
    """
    def __init__(self):
        """
        Creates an instance of the stream time metric.
        """
        super(EvalTime, self).__init__(reset_at='instance',
                                       emit_at='instance',
                                       mode='eval')

    def before_eval(self, strategy: 'BaseStrategy'):
        super().before_eval(strategy)
        self._time.update()

    def __str__(self):
        return "Time_Eval"


@METRICSPLUGIN.register_obj
def timing_metrics(*, iteration=False, epoch=False, whole=False, eval=False):
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.
    :param minibatch: If True, will return a metric able to log the train
        minibatch elapsed time.
    :param epoch: If True, will return a metric able to log the train epoch
        elapsed time.
    :param epoch_running: If True, will return a metric able to log the running
        train epoch elapsed time.
    :param experience: If True, will return a metric able to log the eval
        experience elapsed time.
    :param stream: If True, will return a metric able to log the eval stream
        elapsed time.
    :return: A list of plugin metrics.
    """

    metrics = []
    if iteration:
        metrics.append(IterationTime())

    if epoch:
        metrics.append(EpochTime())

    if whole:
        metrics.append(WholeTime())

    if eval:
        metrics.append(EvalTime())

    return metrics
