import lightning as L
import torch
from torch.nn import functional as F
from tqdm import tqdm

import copy
import logging
import numpy as np
from torch import nn
from defenses.base import DefenseBase


import torchvision
from matplotlib import image as mlt
from torch.utils.data import DataLoader

"""Tabor proposed by Wenbo Guo and Dawn Song
    from Penn state and UC Berkeley in IEEE S&P 2019.

    It is a model inspection backdoor defense
    that inherits :class:`trojanvision.defenses.ModelInspection`.
    (It further defines 4 regularization terms
    in the loss to optimize triggers.)

    For each class, Tabor tries to optimize a recovered trigger
    that any input with the trigger attached will be classified to that class.
    If there is an outlier among all potential triggers, it means the model is poisoned.

    See Also:
        * paper: `TABOR\: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems`_
        * code: https://github.com/UsmannK/TABOR

    Args:
        tabor_hyperparams (list[float]): List of weights for regularization terms.
            Defaults to ``[1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2]``

    .. _TABOR\: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems:
        https://arxiv.org/abs/1908.01763
    """


def repeat_to_batch(x: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    r"""Repeat a single input tensor to a batch.

    Args:
        x (torch.Tensor): The single input tensor to process
            with shape ``(*)``.
        batch_size (int): Batch size. Defaults to ``1``.
    Returns:
        torch.Tensor:
            The batched input tensor with shape ``(batch_size, *)``
    """
    return x.expand([batch_size] + [-1] * x.dim())


class TABOR(DefenseBase):

    def __init__(
        self,
        args,
    ) -> None:
        self.args = args
        self.tabor_hyperparams = [1e-6, 1e-5, 1e-7, 1e-8, 0, 1e-2]

    def setup(
        self,
        clean_train_set,
        clean_test_set,
        poison_train_set,
        poison_test_set,
        model,
        collate_fn,
    ):
        super().setup(
            clean_train_set,
            clean_test_set,
            poison_train_set,
            poison_test_set,
            model,
            collate_fn,
        )
        self.model = model.model
        self.ImageWappermodel = model

    def regularization_loss(
        self, _input: torch.Tensor, _label: torch.Tensor, target: int
    ):
        mark = torch.full((2, 100, 100), 0.5)
        # R1 - Overly large triggers
        mask_l1_norm = mark[-1].abs().sum()
        mask_l2_norm = mark[-1].square().sum()
        mask_r1 = mask_l1_norm + mask_l2_norm

        pattern_tensor: torch.Tensor = (1 - mark[-1]) * mark[:-1]
        pattern_l1_norm = pattern_tensor.abs().sum()
        pattern_l2_norm = pattern_tensor.square().sum()
        pattern_r1 = pattern_l1_norm + pattern_l2_norm

        # R2 - Scattered triggers
        pixel_dif_mask_col = F.mse_loss(
            mark[-1, :-1, :], mark[-1, 1:, :], reduction="sum"
        )
        pixel_dif_mask_row = F.mse_loss(
            mark[-1, :, :-1], mark[-1, :, 1:], reduction="sum"
        )
        mask_r2 = pixel_dif_mask_col + pixel_dif_mask_row

        pixel_dif_pat_col = F.mse_loss(
            pattern_tensor[:, :-1, :], pattern_tensor[:, 1:, :], reduction="sum"
        )
        pixel_dif_pat_row = F.mse_loss(
            pattern_tensor[:, :, :-1], pattern_tensor[:, :, 1:], reduction="sum"
        )
        pattern_r2 = pixel_dif_pat_col + pixel_dif_pat_row

        # R3 - Blocking triggers
        cropped_input_tensor: torch.Tensor = (1 - mark[-1]) * _input
        _cropped_output = self.model(cropped_input_tensor)
        r3 = self.model.criterion(_cropped_output, _label)

        # R4 - Overlaying triggers
        mask_crop_tensor = mark[-1] * mark[:-1]
        mask_crop_tensor = repeat_to_batch(mask_crop_tensor, _label.size(0))
        mask_cropped_output = self.model(mask_crop_tensor)
        r4 = self.model.criterion(mask_cropped_output, target * torch.ones_like(_label))

        loss = (
            self.tabor_hyperparams[0] * mask_r1
            + self.tabor_hyperparams[1] * pattern_r1
            + self.tabor_hyperparams[2] * mask_r2
            + self.tabor_hyperparams[3] * pattern_r2
            + self.tabor_hyperparams[4] * r3
            + self.tabor_hyperparams[5] * r4
        )

        return loss

    def loss(
        self,
        _input: torch.Tensor,
        _label: torch.Tensor,
        target: int,
        trigger_output: torch.Tensor = None,
    ) -> torch.Tensor:
        return super().loss(
            _input, _label, target, trigger_output=trigger_output
        ) + self.regularization_loss(_input, _label, target)

    def train(self):

        retrained_model, results = retraining(
            self.fabric,
            self.clean_train_loader,
            self.clean_test_loader,
            self.poison_test_loader,
            self.model,
            self.args,
            loss_func=self.loss,
        )
        self.args.logger.info(results)

        return results
