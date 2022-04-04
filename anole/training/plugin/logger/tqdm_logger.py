import sys
from typing import List, TYPE_CHECKING
from tqdm import tqdm

from ..metrics import MetricValue
from . import TextLogger
from ...builder import LOGGERPLUGIN

if TYPE_CHECKING:
    from anole.training import BaseStrategy

__all__ = ['TqdmLogger', 'tqdm_logger']


class TqdmLogger(TextLogger):
    """
    Based on tqdm, provide a visualized training process.
    """

    def __init__(self):
        super().__init__(file=sys.stdout)
        self._pbar = None

    def before_training_epoch(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'],
                              **kwargs):
        super().before_training_epoch(strategy, metric_values, **kwargs)
        self._progress.total = strategy.dataset.real_len

    def after_training_epoch(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'],
                             **kwargs):
        self._end_progress()
        super().after_training_epoch(strategy, metric_values, **kwargs)

    def after_training_iteration(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'],
                                 **kwargs):
        self._progress.update()
        self._progress.refresh()
        super().after_training_iteration(strategy, metric_values, **kwargs)

    def before_eval(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'], **kwargs):
        super().before_eval(strategy, metric_values, **kwargs)
        self._progress.total = strategy.dataset.real_len

    def after_eval(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'], **kwargs):
        self._end_progress()
        super().after_eval(strategy, metric_values, **kwargs)

    def after_eval_iteration(self, strategy: 'BaseStrategy', metric_values: List['MetricValue'],
                             **kwargs):
        self._progress.update()
        self._progress.refresh()
        super().after_eval_iteration(strategy, metric_values, **kwargs)

    @property
    def _progress(self):
        if self._pbar is None:
            self._pbar = tqdm(leave=True, position=0, file=sys.stdout)
        return self._pbar

    def _end_progress(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


@LOGGERPLUGIN.register_obj
def tqdm_logger(**kwargs):
    return TqdmLogger(**kwargs)
