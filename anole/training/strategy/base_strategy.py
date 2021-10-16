from abc import abstractmethod
import logging
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from ..builder import STRATEGY

logger = logging.getLogger(__name__)

__all__ = ['BaseStrategy']

class BaseStrategy(object):
    def __init__(self, 
                 model: Module, 
                 optimizer: Optimizer,
                 plugins: List = [],
                 train_epochs: int = 1,
                 eval_every: int = 1, 
                 ckpt_every: int = 1,
                 device='cpu',
                 ):
        self.model = model
        self.optimizer = optimizer
        self.train_epochs = train_epochs
        self.eval_every = eval_every
        self.ckpt_every = ckpt_every
        self.plugins = plugins
        self.device = device
    
    def fit(self, train_dataloader: DataLoader, eval_dataloader: DataLoader):
        
        for epoch in range(1, self.train_epochs + 1): 
            # train phrase
            self.train(train_dataloader)
            # eval phrase
            if epoch % self.eval_every:
                self.eval(eval_dataloader)

                
    def train(self, train_dataloader: DataLoader):
        self.model.train()
        self.before_training_epoch()
        for self.mbatch in train_dataloader:
            self.before_training_iteration()
            self._unpack_train_minibatch()
        
            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self.before_forward()
            self.loss += self.forward()
            self.after_forward()

            # Backward
            self.before_backward()
            self.loss.backward()
            self.after_backward()

            # Optimization step
            self.before_update()
            self.optimizer.step()
            self.after_update()

            self.after_training_iteration()
        self.after_training_epoch()
    
    def eval(self, eval_dataloader: DataLoader):
        self.model.eval()
        self.before_eval_epoch()
        with torch.no_grad():
            for self.mbatch in eval_dataloader:
                self.before_eval_iteration()
                self._unpack_eval_minibatch()

                self.before_eval_forward()
                self.loss = self.forward()
                self.after_eval_forward()

                self.after_eval_iteration()
        self.after_eval_epoch()
    
    @abstractmethod
    def _unpack_train_minibatch(self):
        pass
    
    @abstractmethod
    def _unpack_eval_minibatch(self):
        pass
    
    @abstractmethod
    def forward(self):
        pass
    
    # training plugins
    def before_training_epoch(self):
        for p in self.plugins:
            p.before_training_epoch(self)
            
    def after_training_epoch(self):
        for p in self.plugins:
            p.after_training_epoch(self)
            
    def before_training_iteration(self):
        for p in self.plugins:
            p.before_training_iteration(self)
            
    def after_training_iteration(self):
        for p in self.plugins:
            p.after_training_iteration(self)
            
    def before_forward(self):
        for p in self.plugins:
            p.before_forward(self)
            
    def after_forward(self):
        for p in self.plugins:
            p.after_forward(self)
            
    def before_backward(self):
        for p in self.plugins:
            p.before_backward(self)
            
    def after_backward(self):
        for p in self.plugins:
            p.after_backward(self)
            
    def before_update(self):
        for p in self.plugins:
            p.before_update(self)
            
    def after_update(self):
        for p in self.plugins:
            p.after_update(self)
            
    # eval plugins           
    def before_eval_epoch(self):
        for p in self.plugins:
            p.before_eval_epoch(self)
            
            
    def after_eval_epoch(self):
        for p in self.plugins:
            p.after_eval_epoch(self)
            
            
    def before_eval_iteration(self):
        for p in self.plugins:
            p.before_eval_iteration(self)
            
    def after_eval_iteration(self):
        for p in self.plugins:
            p.after_eval_iteration(self)
            
    def before_eval_forward(self):
        for p in self.plugins:
            p.before_eval_forward(self)
            
    def after_eval_forward(self):
        for p in self.plugins:
            p.after_eval_forward(self)
        
    
@STRATEGY.register_obj
def base_strategy(model, optimizer, **kwargs):
    return BaseStrategy(model, optimizer, **kwargs)
        

