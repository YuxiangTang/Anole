import os
import sys
import argparse
import logging

logger = logging.getLogger(__name__)

import hydra
from omegaconf import OmegaConf
from torch import optim

from anole.utils import build_from_cfg, load_ckpt, init_dataloader
from anole.training.plugin import EvaluationPlugin
from anole.benchmarks import DATASET
from anole.model import PIPELINE
from anole.training import STRATEGY, LOGGERPLUGIN, METRICSPLUGIN
from anole.model import LOSS

sys.path.append("anole")
home_path = os.getcwd()


@hydra.main(config_path="./config")
def main(config):
    global logger
    # --- CONFIG
    cfg = OmegaConf.to_yaml(config)
    cfg = OmegaConf.structured(cfg)
    logger.info(cfg)

    # --- build dataloader
    train_dataset, eval_dataset = build_from_cfg(cfg.dataset.name, cfg.dataset.params, DATASET)
    train_dataloader = init_dataloader(train_dataset, **cfg.dataset.train_loader)
    eval_dataloader = init_dataloader(eval_dataset, **cfg.dataset.eval_loader)

    # --- build model
    model = build_from_cfg(cfg.model.name, cfg.model, PIPELINE).to(cfg.device)

    # --- build criterion
    criterion = build_from_cfg(cfg.criterion.name, cfg.criterion.params, LOSS).to(cfg.device)

    # --- build optimizer
    optimizer = getattr(optim, cfg.optimizer.name)(model.parameters(), **cfg.optimizer.params)

    # --- build lr_scheduler
    lr_scheduler = getattr(optim.lr_scheduler, cfg.lr_scheduler.name)(optimizer, **cfg.lr_scheduler.params)

    # --- load form ckpt ---
    model, optimizer, lr_scheduler = load_ckpt(model, optimizer, lr_scheduler, f"{home_path}/{cfg.checkpoint_path}")

    # --- buld plugins (metric and logger) & training strategy
    metrics = []
    logger = []
    if hasattr(cfg, 'plugin'):
        if hasattr(cfg.plugin, 'metric'):
            metrics = build_from_cfg(cfg.plugin.metric.name, cfg.plugin.metric.params, METRICSPLUGIN)
        if hasattr(cfg.plugin, 'logger'):
            logger = build_from_cfg(cfg.plugin.logger.name, cfg.plugin.logger.params, LOGGERPLUGIN)
    plugins = EvaluationPlugin(*metrics, loggers=logger)

    strategy = build_from_cfg(
        cfg.strategy.name, {
            'model': model,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'criterion': criterion,
            'plugins': plugins,
            'device': cfg.device,
            **cfg.strategy.params
        }, STRATEGY)

    if cfg.mode == 'train':
        # --- training
        strategy.fit(train_dataloader, eval_dataloader)
    else:
        # --- evaluation
        strategy.eval(eval_dataloader)


if __name__ == '__main__':
    main()
