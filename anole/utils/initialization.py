import os
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def init_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 10,
    drop_last: bool = True,
    persistent_workers: bool = False,
    shuffle: bool = False,
    **kwargs,
):
    setattr(dataset, 'real_len', len(dataset) // batch_size)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      prefetch_factor=prefetch_factor,
                      persistent_workers=persistent_workers,
                      drop_last=drop_last,
                      shuffle=shuffle,
                      **kwargs)


def load_ckpt(model, optimizer, lr_scheduler, checkpoint_path):
    """
    load Checkpoint here.
    """
    if not os.path.exists(checkpoint_path):
        logger.info("checkpoint is unavailable.")
        return model, optimizer, lr_scheduler

    ckpt = torch.load(checkpoint_path)
    ckpt_model = ckpt['model']
    ckpt_opt = ckpt['optimizer']
    ckpt_epoch = ckpt['epoch']

    logger.info(f'Load from epoch: {ckpt_epoch}.')
    for _ in range(ckpt_epoch):
        lr_scheduler.step()

    model.load_state_dict(ckpt_model)
    optimizer.load_state_dict(ckpt_opt)
    logger.info('Load from checkpoint successfully.')

    return model, optimizer, lr_scheduler
