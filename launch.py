import os
import sys
import argparse
import logging

from omegaconf import OmegaConf
import torch

from anole.utils import build_from_cfg
from anole.benchmarks import DATASET
from anole.model import PIPELINE
from anole.training import STRATEGY

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
    dataloader = build_from_cfg(cfg.dataset.name, cfg.dataset.params, DATASET)
    
    # --- build model
    model = build_from_cfg(cfg.model.pipeline.name, cfg.model, PIPELINE)
    
    # --- build optimizer
    optimizer = torch.optim.Adam([{'params':model.parameters() , 'lr':1e-5, 'weight_decay':5e-5}])
    
    # --- buld training strategy & TODO: plugins
    # plugins = A List
    strategy = build_from_cfg(cfg.strategy.name, 
                              {'model':model, 'optimizer': optimizer, **cfg.strategy.params}, 
                              STRATEGY)
    
    
    
    # --- TODO: learning rate scheduling

    # --- process from config
    
    # --- training
    strategy.fit(dataloader, dataloader)

if __name__ == '__main__':
    main()