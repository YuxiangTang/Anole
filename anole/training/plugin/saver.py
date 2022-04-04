import os

import torch
from typing import TYPE_CHECKING

from .base_plugin import BasePlugin

if TYPE_CHECKING:
    from ..strategy.base_strategy import BaseStrategy

__all__ = ['Saver']


class Saver(BasePlugin):
    """
    Save the checkpoint.
    """

    def __init__(self, exp_name, save_path):
        """
        :param exp_name: the name of this checkpoint.
        :param save_path: the path that save the checkpoint.
        """
        super().__init__()
        # train
        self.exp_name = exp_name
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def save_ckpt(self, strategy: 'BaseStrategy', mode: str):
        assert mode in ['schedule', 'best']
        model = strategy.model
        optimizer = strategy.optimizer
        total_epoch = strategy.clock.train_epochs
        total_iteration = strategy.clock.total_iterations
        dataset_name = strategy.dataset.dataset_name
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': total_epoch,
            'iteration': total_iteration,
        }
        if mode == 'best':
            torch.save(ckpt, f'{self.save_path}/{self.exp_name}_{dataset_name}_best.pth')
        else:
            torch.save(ckpt, f'{self.save_path}/{self.exp_name}_{dataset_name}_{total_epoch}.pth')
