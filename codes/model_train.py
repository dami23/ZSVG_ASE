import torch
import fire
import numpy as np
from functools import partial
import warnings
warnings.filterwarnings('ignore')
import os, pdb

from data_loader import get_data
from mdl import get_default_net
from loss import get_default_loss
from evaluator import get_default_eval
from utils import Learner, synchronize
from extended_config import (cfg as conf, key_maps, CN, update_from_dict)

def learner_init(uid: str, cfg: CN) -> Learner:
    device = torch.device('cuda')
    data = get_data(cfg)

    # Ugly hack because I wanted ratios, scales
    # in fractional formats
    if type(cfg['ratios']) != list:
        ratios = eval(cfg['ratios'], {})
    else:
        ratios = cfg['ratios']
    if type(cfg['scales']) != list:
        scales = cfg['scale_factor'] * np.array(eval(cfg['scales'], {}))
    else:
        scales = cfg['scale_factor'] * np.array(cfg['scales'])
    
    num_anchors = len(ratios) * len(scales)
    mdl = get_default_net(num_anchors=num_anchors, cfg=cfg)
    mdl.to(device)

    if cfg.do_dist:
        mdl = torch.nn.parallel.DistributedDataParallel(
            mdl, device_ids=[0],
            output_device=0, broadcast_buffers=True,
            find_unused_parameters=True)
    
    elif not cfg.do_dist and cfg.num_gpus:
        # Use data parallel
        mdl = torch.nn.DataParallel(mdl)

    loss_fn = get_default_loss(ratios, scales, cfg)
    loss_fn.to(device)

    eval_fn = get_default_eval(ratios, scales, cfg)

    opt_fn = partial(torch.optim.Adam, betas=(0.9, 0.99))

    learn = Learner(uid=uid, data=data, mdl=mdl, loss_fn=loss_fn,
                    opt_fn=opt_fn, eval_fn=eval_fn, device=device, cfg=cfg)
    return learn

def main_dist(uid: str, **kwargs):
    cfg = conf
    num_gpus = torch.cuda.device_count()
    cfg.num_gpus = num_gpus

    if num_gpus > 1:
        if 'local_rank' in kwargs:
            # We are doing distributed parallel
            cfg.do_dist = True
            torch.cuda.set_device(0)  #kwargs['local_rank']
            torch.distributed.init_process_group(
                backend="nccl", init_method="tcp://127.0.0.1:12345", rank=0, world_size=4
            )
            synchronize()
        else:
            # We are doing data parallel
            cfg.do_dist = False

    # Update the config file depending on the command line args
    cfg = update_from_dict(cfg, kwargs, key_maps)

    # Freeze the cfg, can no longer be changed
    cfg.freeze()

    # Initialize learner
    learn = learner_init(uid, cfg)

    # Train or Test
    if not (cfg.only_val or cfg.only_test):
        learn.fit(epochs=cfg.epochs, lr=cfg.lr)
    else:
        if cfg.only_val:
            learn.testing(learn.data.valid_dl)
        
        if cfg.only_test:
            if cfg.ds_to_use == 'refcoco' or cfg.ds_to_use == 'refcoco+':
                learn.testing(learn.data.testA_dl)
                learn.testing(learn.data.testB_dl)

            else:
                learn.testing(learn.data.test_dl)
        
if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    fire.Fire(main_dist)
