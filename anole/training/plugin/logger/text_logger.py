from typing import List, Tuple, Type, TYPE_CHECKING

import torch
import logging

from .base_logger import BaseLogger
from ..metrics import MetricValue
from ...builder import LOGGERPLUGIN

if TYPE_CHECKING:
    from anole.training import BaseStrategy

__all__ = ['TextLogger', 'text_logger']

logger = logging.getLogger(__name__)


class TextLogger(BaseLogger):
    """
    The `TextLogger` class provides logging facilities
    printed to a user specified file. The logger writes
    metric results after each training epoch, evaluation
    experience and at the end of the entire evaluation stream.
    .. note::
        To avoid an excessive amount of printed lines,
        this logger will **not** print results after
        each iteration. If the user is monitoring
        metrics which emit results after each minibatch
        (e.g., `MinibatchAccuracy`), only the last recorded
        value of such metrics will be reported at the end
        of the epoch.
    .. note::
        Since this logger works on the standard output,
        metrics producing images or more complex visualizations
        will be converted to a textual format suitable for
        console printing. You may want to add more loggers
        to your `EvaluationPlugin` to better support
        different formats.
    """

    def __init__(self):
        """
        Creates an instance of `TextLogger` class.
        :param file: destination file to which print metrics
            (default=sys.stdout).
        """
        super().__init__()
        self.metric_vals = {}

        self.epoch = 0
        self.total_epoch = 0

        self.train_iterations = 0
        self.total_iterations = 0
        self.instance_iteration = 0

        self.is_training = False
        global logger

    def log_single_metric(self, name, value, x_plot) -> None:
        self.metric_vals[name] = (name, x_plot, value)

    def _start(self, strategy: 'BaseStrategy'):
        action_name = 'training' if strategy.is_training else 'eval'
        name = strategy.dataset.dataset_name
        logger.info('-- Starting {} on {}--'.format(action_name, name))

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return '\n' + str(m_val)
        elif isinstance(m_val, float):
            return f'{m_val:.4f}'
        else:
            return str(m_val)

    def print_current_metrics(self):
        if len(self.metric_vals) == 0:
            return
        sorted_vals = self.metric_vals.values(
        )  # sorted(self.metric_vals.values(), key=lambda x: x[0])
        msg = "Epoch:{}, Step:{} [{}/{}]".format(self.epoch + 1, self.total_iterations + 1,
                                                 self.train_iterations + 1, self.instance_iteration)
        if self.is_training:
            msg += ", TRAIN: "
        else:
            msg += ", EVAL: "

        msg_eval = ""
        for name, x, val in sorted_vals:
            val = self._val_to_str(val)
            if name.split('_')[-1] in ["Mean", "Med", "Tri", "T25", "L25"]:
                msg_eval += f'{name} = {val}, '
            else:
                msg += f'{name} = {val}, '

        logger.info(msg)
        if msg_eval != "":
            logger.info(msg_eval)

    # *************** Training Phase ***************
    def before_training(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'],
                        **kwargs):
        super().before_training(strategy, metric_values, **kwargs)
        self.total_epoch = strategy.train_epochs
        # logger.info('-- Start of training phase --')

    def after_training(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'],
                       **kwargs):
        super().after_training(strategy, metric_values, **kwargs)
        # logger.info('-- End of training phase --')

    def before_training_epoch(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'],
                              **kwargs):
        super().before_training_epoch(strategy, metric_values, **kwargs)
        self.epoch = strategy.clock.train_epochs
        self.instance_iteration = strategy.dataset.real_len
        self.is_training = True
        self._start(strategy)

    def after_training_epoch(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'],
                             **kwargs):
        super().after_training_epoch(strategy, metric_values, **kwargs)
        self.print_current_metrics()
        logger.info(f'Epoch {strategy.clock.train_epochs + 1} ended.')
        self.metric_vals = {}

    def before_training_iteration(self, strategy: 'BaseStrategy',
                                  metric_values: List['MetricValue'], **kwargs):
        super().before_training_iteration(strategy, metric_values, **kwargs)
        self.train_iterations = strategy.clock.train_iterations
        self.total_iterations = strategy.clock.total_iterations

    def after_training_iteration(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'],
                                 **kwargs):
        super().after_training_iteration(strategy, metric_values, **kwargs)
        if (self.train_iterations + 1) % 10 == 0:
            self.print_current_metrics()
        self.metric_vals = {}

    # *************** Eval Phase ***************
    def before_eval(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'], **kwargs):
        super().before_eval(strategy, metric_values, **kwargs)
        self.is_training = False
        # logger.info('-- >> Start of eval phase << --')

    def after_eval(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'], **kwargs):
        super().after_eval(strategy, metric_values, **kwargs)
        self.print_current_metrics()
        # logger.info('-- >> End of eval phase << --')
        self.metric_vals = {}


@LOGGERPLUGIN.register_obj
def text_logger(**kwargs):
    return TextLogger(**kwargs)
