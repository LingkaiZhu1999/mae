# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

IMAGENET_TRAIN_SAMPLES = 1281167
NUM_CLASSES = 1000

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    run=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    steps_per_epoch = max(1, IMAGENET_TRAIN_SAMPLES // args.batch_size)

    optimizer.zero_grad()

    for data_iter_step, (samples, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header, iterable_len=steps_per_epoch)):
        if data_iter_step >= steps_per_epoch:
            break

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / steps_per_epoch + epoch, args)

        samples = samples.to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        #     loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        if args.bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        else:
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=update_grad)
            if update_grad:
                optimizer.zero_grad()
        else:
            loss.backward()
            if update_grad:
                optimizer.step()
                optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if run is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / steps_per_epoch + epoch) * 1000)
            run.log({
                'train_loss': loss_value_reduce,
                'lr': lr,
                'epoch_1000x': epoch_1000x,
            }, step=epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
