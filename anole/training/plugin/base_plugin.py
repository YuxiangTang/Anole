from ..strategy import BaseStrategy

class BasePlugin(object):
    """
    Base class for strategy plugins. Implements all the callbacks required
    by the BaseStrategy with an empty function. Subclasses must override
    the callbacks.
    """

    def __init__(self):
        super().__init__()
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
            
    def before_update(self, strategy: 'BaseStrategy'):
        pass
            
    def after_update(self, strategy: 'BaseStrategy'):
        pass
            
    # eval plugins           
    def before_eval_epoch(self, strategy: 'BaseStrategy'):
        pass
            
    def after_eval_epoch(self, strategy: 'BaseStrategy'):
        pass
                
    def before_eval_iteration(self, strategy: 'BaseStrategy'):
        pass
            
    def after_eval_iteration(self, strategy: 'BaseStrategy'):
        pass
            
    def before_eval_forward(self, strategy: 'BaseStrategy'):
        pass
            
    def after_eval_forward(self, strategy: 'BaseStrategy'):
        pass