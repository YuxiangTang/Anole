from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..strategy import BaseStrategy


class BasePlugin(ABC):
    """
    Base class for strategy plugins. Implements all the callbacks required
    by the BaseStrategy with an empty function. Subclasses must override
    the callbacks.
    """
    def __init__(self):
        super().__init__()
        pass

    # *************** Training Phase ***************
    def before_training(self, strategy: 'BaseStrategy'):
        pass

    def after_training(self, strategy: 'BaseStrategy'):
        pass

    def before_training_epoch(self, strategy: 'BaseStrategy'):
        pass

    def after_training_epoch(self, strategy: 'BaseStrategy'):
        pass

    def before_training_iteration(self, strategy: 'BaseStrategy'):
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy'):
        pass

    def before_forward(self, strategy: 'BaseStrategy'):
        pass

    def after_forward(self, strategy: 'BaseStrategy'):
        pass

    def before_backward(self, strategy: 'BaseStrategy'):
        pass

    def after_backward(self, strategy: 'BaseStrategy'):
        pass

    # *************** Eval Phase ***************
    def before_eval(self, strategy: 'BaseStrategy'):
        pass

    def after_eval(self, strategy: 'BaseStrategy'):
        pass

    def before_eval_iteration(self, strategy: 'BaseStrategy'):
        pass

    def after_eval_iteration(self, strategy: 'BaseStrategy'):
        pass
