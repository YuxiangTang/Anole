import os
import sys
import argparse
import logging

from omegaconf import OmegaConf
import torch
from anole.training import plugin

from anole.utils import build_from_cfg, init_model, init_dataloader
from anole.training.plugin import EvaluationPlugin
from anole.benchmarks import DATASET
from anole.model import PIPELINE
from anole.training import STRATEGY, LOGGERPLUGIN, METRICSPLUGIN
from anole.model import LOSS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args.cfg


def main():
    cfg_path = parse_args()

    # --- CONFIG
    cfg = OmegaConf.to_yaml(OmegaConf.load(cfg_path))
    cfg = OmegaConf.structured(cfg)
    print(cfg)

    # --- build dataloader
    train_dataset, eval_dataset = build_from_cfg(cfg.dataset.name,
                                                 cfg.dataset.params, DATASET)
    train_dataloader = init_dataloader(train_dataset,
                                       **cfg.dataset.train_loader)
    eval_dataloader = init_dataloader(eval_dataset, **cfg.dataset.eval_loader)

    # --- build model
    model = build_from_cfg(cfg.model.name, cfg.model, PIPELINE)
    print(model)
    model = init_model(model, **cfg.model.params)

    # --- build criterion
    criterion = build_from_cfg(cfg.criterion.name, cfg.criterion.params, LOSS)

    # --- build optimizer
    optimizer = torch.optim.Adam([{
        'params': model.parameters(),
        'lr': 1e-4,
        # 'weight_decay': 5e-5
    }])

    # --- buld plugins (metric and logger) & training strategy
    metrics = []
    logger = []
    if hasattr(cfg, 'plugin'):
        if hasattr(cfg.plugin, 'metric'):
            metrics = build_from_cfg(cfg.plugin.metric.name,
                                     cfg.plugin.metric.params, METRICSPLUGIN)
        if hasattr(cfg.plugin, 'logger'):
            logger = build_from_cfg(cfg.plugin.logger.name,
                                    cfg.plugin.logger.params, LOGGERPLUGIN)
    plugins = EvaluationPlugin(*metrics, loggers=logger)

    strategy = build_from_cfg(
        cfg.strategy.name, {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'plugins': plugins,
            **cfg.strategy.params
        }, STRATEGY)

    # --- TODO: learning rate scheduling

    if cfg.mode == 'train':
        # --- training
        strategy.fit(train_dataloader, eval_dataloader)
    else:
        # --- evaluation
        strategy.eval(eval_dataloader)


if __name__ == '__main__':
    main()
