from abc import abstractmethod
import logging
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from ..plugin import default_logger, Clock

if TYPE_CHECKING:
    from ..plugin import BasePlugin

logger = logging.getLogger(__name__)

__all__ = ['BaseStrategy']


class BaseStrategy(object):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: Module,
        plugins: 'BasePlugin' = None,
        train_epochs: int = 1,
        iterations_interval: int = 10,
        eval_every: int = 1,
        ckpt_every: int = 1,
        device: str = 'cuda:0',
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self._criterion = criterion.to(device)
        self.train_epochs = train_epochs
        self.eval_every = eval_every
        self.ckpt_every = ckpt_every
        self.plugins = []
        self.device = device

        if plugins is None:
            self.plugins.append(default_logger)
        else:
            self.plugins.append(plugins)

        self.clock = Clock(iterations_interval)
        self.plugins.append(self.clock)

    def fit(self, train_dataloader: DataLoader, eval_dataloader: DataLoader):
        self.before_training()
        if type(train_dataloader) is not list:
            train_dataloader = [train_dataloader]

        for epoch in range(1, self.train_epochs + 1):
            # train phase
            for loader in train_dataloader:
                self.train(loader)

            # eval phase
            if epoch % self.eval_every == 0:
                self.eval(eval_dataloader)
        self.after_training()

    def train(self, train_dataloader: DataLoader):
        self.dataset = train_dataloader.dataset
        self.is_training = True
        self.model.train()
        self.before_training_epoch()
        for it, self.train_batch in enumerate(train_dataloader):
            self.before_training_iteration()
            self._unpack_train_minibatch()

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self.before_forward()
            self.pred = self.forward()
            self.after_forward()

            # Backward
            self.loss = self.criterion()

            self.before_backward()
            self.loss.backward()
            self.after_backward()

            # Optimization step
            self.optimizer.step()

            self.after_training_iteration()
        self.after_training_epoch()

    def eval(self, eval_dataloader: DataLoader):
        if type(eval_dataloader) is not list:
            eval_dataloader = [eval_dataloader]
        self.is_training = False
        self.model.eval()
        self.before_eval()
        with torch.no_grad():
            for loader in eval_dataloader:
                self.dataset = loader.dataset
                for it, self.eval_batch in enumerate(loader):
                    self.before_eval_iteration()
                    self._unpack_eval_minibatch()

                    self.pred = self.forward()
                    self.loss = self.criterion()

                    self.after_eval_iteration()
        self.after_eval()

    @abstractmethod
    def _unpack_train_minibatch(self):
        pass

    @abstractmethod
    def _unpack_eval_minibatch(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def criterion(self):
        pass

    # *************** Training Phase ***************
    def before_training(self, **kwargs):
        for p in self.plugins:
            p.before_training(self, **kwargs)

    def after_training(self, **kwargs):
        for p in self.plugins:
            p.after_training(self, **kwargs)

    def before_training_epoch(self, **kwargs):
        for p in self.plugins:
            p.before_training_epoch(self, **kwargs)

    def after_training_epoch(self, **kwargs):
        for p in self.plugins:
            p.after_training_epoch(self, **kwargs)

    def before_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_training_iteration(self, **kwargs)

    def after_training_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_training_iteration(self, **kwargs)

    def before_forward(self, **kwargs):
        for p in self.plugins:
            p.before_forward(self, **kwargs)

    def after_forward(self, **kwargs):
        for p in self.plugins:
            p.after_forward(self, **kwargs)

    def before_backward(self, **kwargs):
        for p in self.plugins:
            p.before_backward(self, **kwargs)

    def after_backward(self, **kwargs):
        for p in self.plugins:
            p.after_backward(self, **kwargs)

    # *************** Eval Phase ***************
    def before_eval(self, **kwargs):
        for p in self.plugins:
            p.before_eval(self, **kwargs)

    def after_eval(self, **kwargs):
        for p in self.plugins:
            p.after_eval(self, **kwargs)

    def before_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.before_eval_iteration(self, **kwargs)

    def after_eval_iteration(self, **kwargs):
        for p in self.plugins:
            p.after_eval_iteration(self, **kwargs)
