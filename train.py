import os
import sys
import logging
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader
from detectron2.utils.env import seed_all_rng

import utils
import models
import checkpoint


_PORT = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
_DIST_URL = f"tcp://127.0.0.1:{_PORT}"
_DEFAULT_TIMEOUT = timedelta(minutes=30)


def train(cfg):
    n_gpu = len(cfg.cuda.indices)
    cfg.output_dir = cfg.output_dir

    if n_gpu > 1:
        mp.spawn(
            _dist_train_worker,
            nprocs=n_gpu,
            args=(
                _train_for_object_detection,
                n_gpu,
                n_gpu,
                0,
                _DIST_URL,
                (cfg,),
                _DEFAULT_TIMEOUT,
            ),
            daemon=False,
        )
    else:
        _train_for_object_detection(cfg)


def _dist_train_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
    timeout,
):
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    dist.init_process_group(
        backend="NCCL",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    comm.synchronize()
    main_func(*args)


def _train_for_object_detection(cfg):

    output_path = Path(cfg.output_dir)

    # Seed random number generator.
    seed_all_rng(cfg.seed + comm.get_rank())
    torch.manual_seed(cfg.seed)

    if comm.is_main_process():
        logging.info(f"Output path: {output_path}")

    # Build data loader.
    if hasattr(cfg.setting, 'vision_network'):
        vision_task = cfg.setting.vision_network.task
        vision_model = cfg.setting.vision_network.model
    else:  # dummy for dataloader
        vision_task = 'detection'
        vision_model = 'faster_rcnn_X_101_32x8d_FPN_3x'
    _od_cfg = utils.get_od_cfg(vision_task, vision_model)
    _od_cfg.SOLVER.IMS_PER_BATCH = cfg.batch_size
    dataloader = build_detection_train_loader(_od_cfg)

    # Build end-to-end model.
    end2end_network = models.EndToEndNetwork(cfg)

    # Load on GPU.
    end2end_network.cuda()

    # Set mode as training.
    end2end_network.train()

    filter_train_flag = hasattr(cfg.setting, 'filtering_network') and cfg.setting.filtering_network.train
    estimator_train_flag = hasattr(cfg.setting, 'rate_estimator') and cfg.setting.rate_estimator.train
    estimator_restore_flag = hasattr(cfg.setting, 'rate_estimator') and cfg.setting.rate_estimator.pretrained

    # Build optimizers.
    if filter_train_flag:
        filtering_cfg = cfg.setting.filtering_network
        filtering_params = list(end2end_network.filtering_network.parameters())
        filtering_optimizer, filtering_optimizer_scheduler = _create_optimizer(
            filtering_params,
            filtering_cfg.optimizer.name,
            filtering_cfg.optimizer.scheduler,
            filtering_cfg.optimizer.learning_rate,
            cfg.total_step,
            filtering_cfg.optimizer.final_lr_ratio)
        if comm.is_main_process():
            filtering_ckpt = checkpoint.Checkpoint(output_path / 'filtering_network')

    if estimator_train_flag:
        estimator_cfg = cfg.setting.rate_estimator
        estimator_params = list(end2end_network.rate_estimator.parameters())
        estimator_optimizer, estimator_optimizer_scheduler = _create_optimizer(
            estimator_params,
            estimator_cfg.optimizer.name,
            estimator_cfg.optimizer.scheduler,
            estimator_cfg.optimizer.learning_rate,
            cfg.total_step,
            estimator_cfg.optimizer.final_lr_ratio)
        if comm.is_main_process():
            estimator_ckpt = checkpoint.Checkpoint(output_path / 'rate_estimator')

    if estimator_restore_flag:
        _ckpt_path = Path(cfg.setting.rate_estimator.pretrained)
        _ckpt = checkpoint.Checkpoint(_ckpt_path / 'train' / 'rate_estimator')
        _ckpt.load(end2end_network.rate_estimator, cfg.setting.rate_estimator.pretrained_step)

    # Distributed.
    distributed = comm.get_world_size() > 1
    if distributed:
        end2end_network = DistributedDataParallel(
            end2end_network,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    # Create summary writer.
    if comm.is_main_process():
        writer = SummaryWriter(output_path)

    # Run training loop.
    logging.info("Start training.")
    end_step = cfg.total_step

    for data, step in zip(dataloader, range(1, end_step + 1)):

        # Prepare control input.
        if cfg.setting.control_input == 'none':
            control_input = None
        elif cfg.setting.control_input == 'random':
            control_input = np.random.rand(cfg.batch_size // comm.get_world_size())
        else:
            control_input = np.ones((cfg.batch_size,)) * cfg.setting.control_input

        # Forward pass.
        outs = end2end_network(data, control_input=control_input)

        # Backpropagation.
        if filter_train_flag:
            filtering_optimizer.zero_grad()
            loss_rd = outs['loss_r'] + outs['loss_d']
            loss_rd.backward(retain_graph=estimator_train_flag)
            filtering_optimizer.step()
            filtering_optimizer_scheduler.step()

        if estimator_train_flag:
            estimator_optimizer.zero_grad()
            loss_aux = outs['loss_aux']
            loss_aux.backward(inputs=estimator_params)
            estimator_optimizer.step()
            estimator_optimizer_scheduler.step()

        # Write on tensorboard.
        if comm.is_main_process():
            scalars = dict()
            for k, v in outs.items():
                if len(v.shape) == 0:
                    scalars[k] = v
            scalars = comm.reduce_dict(scalars)
            if filter_train_flag:
                scalars.update({'learning_rate/filtering_network': filtering_optimizer_scheduler.get_last_lr()[0]})
            if estimator_train_flag:
                scalars.update({'learning_rate/rate_estimator': estimator_optimizer_scheduler.get_last_lr()[0]})
            writer.add_scalars('train', scalars, step)

            if step % 100 == 0:
                info_msg = f"step: {step:6} "
                for k, v in scalars.items():
                    info_msg += f"| {k}: {v:7.4f} "
                logging.info(info_msg)

                if step % cfg.saving_period == 0:
                    if distributed:
                        if filter_train_flag:
                            network = end2end_network.module.filtering_network
                            filtering_ckpt.save(network, step)
                        if estimator_train_flag:
                            network = end2end_network.module.rate_estimator
                            estimator_ckpt.save(network, step)
                    else:
                        if filter_train_flag:
                            network = end2end_network.filtering_network
                            filtering_ckpt.save(network, step)
                        if estimator_train_flag:
                            network = end2end_network.rate_estimator
                            estimator_ckpt.save(network, step)


def _create_optimizer(parameters, optim_name, scheduler_name, initial_lr, total_steps, final_rate=.1):
    if optim_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=initial_lr, momentum=0.9)
    elif optim_name == 'adam':
        optimizer = optim.Adam(parameters, lr=initial_lr)
    else:
        raise NotImplemented("Currently supported optimizers are 'adam' and 'sgd'.")

    if scheduler_name == 'constant':
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_name == 'exponential':
        gamma = final_rate ** (1 / total_steps)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise NotImplemented("Currently supported schedulers are 'constant' and 'exponential'.")
    return optimizer, scheduler