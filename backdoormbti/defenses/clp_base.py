"""
Data-free Backdoor Removal based on Channel Lipschitzness
This file is for clp_base denfense:
github link : https://github.com/rkteddy/channel-Lipschitzness-based-pruning.

@article{zheng2022data,
  title={Data-free Backdoor Removal based on Channel Lipschitzness},
  author={Zheng, Runkai and Tang, Rongjun and Li, Jianze and Liu, Li},
  journal={arXiv preprint arXiv:2208.03111},
  year={2022}
}
"""

import copy
import logging

import numpy as np
import torch
from torch import nn
from defenses.base import DefenseBase


def CLP_prune(net, u):
    params = net.state_dict()
    # conv = None
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = (
                    conv.weight[idx].reshape(conv.weight.shape[1], -1)
                    * (weight[idx] / std[idx]).abs()
                )
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(
                channel_lips > channel_lips.mean() + u * channel_lips.std()
            )[0]

            params[name + ".weight"][index] = 0
            params[name + ".bias"][index] = 0

        # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)


class ClpBase(DefenseBase):
    def __init__(self, args) -> None:
        self.args = args

    def train(self):
        logger = logging.getLogger("defense")
        self.model.eval()
        for u in np.arange(self.args.u_min, self.args.u_max, 0.5):
            logger.info(f"current process u == {u}")
            model_copy = copy.deepcopy(self.model)
            model_copy.eval()
            CLP_prune(model_copy, u)
            results = super().eval()
            logger.info(f"when u = {u}, acc,asr, ra:{results}")
        return results
