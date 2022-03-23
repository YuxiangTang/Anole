from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.utils.data import DataLoader


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


def init_model(model, **kwargs):
    """
    TODO: load pretrained model
    Hint: Note the keys of the model
    """
    return model
