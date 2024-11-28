"""
This file is modified based on the following source:
link : https://github.com/bboylyg/ABL.
The defense method is called abl.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. abl defense:
        a. pre-train model
        b. isolate the special data(loss is low) as backdoor data
        c. unlearn the backdoor data and learn the remaining data
    4. test the result and get ASR, ACC, RC
"""

import os
import sys


import torch

import torch.nn.functional as F


from defenses.abl_base import Abl_Base

from utils.data import get_dataloader
from utils.model import load_model


sys.path.append("../")
sys.path.append(os.getcwd())

import logging


def all_acc(
    preds: torch.Tensor,
    labels: torch.Tensor,
):
    if len(preds) == 0 or len(labels) == 0:
        logging.warning("zero len array in func all_acc(), return None!")
        return None
    return preds.eq(labels).sum().item() / len(preds)


def given_dataloader_test(
    args,
    model,
    test_dataloader,
    criterion,
    non_blocking: bool = False,
    device="cpu",
    verbose: int = 0,
):
    model.to(device, non_blocking=non_blocking)
    model.eval()
    metrics = {
        "test_correct": 0,
        "test_loss_sum_over_batch": 0,
        "test_total": 0,
    }
    criterion = criterion.to(device, non_blocking=non_blocking)

    if verbose == 1:
        batch_predict_list, batch_label_list = [], []

    with torch.no_grad():

        for batch_idx, (data, target, *additional_info) in enumerate(test_dataloader):
            data = args.tokenizer(
                data, padding=True, truncation=True, return_tensors="pt"
            )
            data["labels"] = target
            data = data.to(args.device)
            target = target.to(args.device)
            ret = model(**data)
            loss = ret.loss
            # for efficiency
            probs = F.softmax(ret.logits, dim=1)

            # get predictions
            predictions = torch.argmax(probs, dim=1)

            _, predicted = torch.max(probs, -1)
            correct = predicted.eq(target).sum()

            if verbose == 1:
                batch_predict_list.append(predicted.detach().clone().cpu())
                batch_label_list.append(target.detach().clone().cpu())

            metrics["test_correct"] += correct.item()
            metrics["test_loss_sum_over_batch"] += loss.item()
            metrics["test_total"] += target.size(0)

    metrics["test_loss_avg_over_batch"] = metrics["test_loss_sum_over_batch"] / len(
        test_dataloader
    )
    metrics["test_acc"] = metrics["test_correct"] / metrics["test_total"]

    if verbose == 0:
        return metrics, None, None
    elif verbose == 1:
        return metrics, torch.cat(batch_predict_list), torch.cat(batch_label_list)


class Abl(Abl_Base):
    def __init__(self, args) -> None:
        self.args = args

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
            clean_train_set=clean_train_set,
            clean_test_set=clean_test_set,
            poison_train_set=poison_train_set,
            poison_test_set=poison_test_set,
            model=model,
            collate_fn=collate_fn,
        )

    def train(self):
        return super().train()

    def train_step(self, args, train_loader, model_ascent, optimizer, criterion, epoch):
        """Pretrain the model with raw data for each step
        args:
            Contains default parameters
        train_loader:
            the dataloader of train data
        model_ascent:
            the initial model
        optimizer:
            optimizer during the pretrain process
        criterion:
            criterion during the pretrain process
        epoch:
            current epoch
        """
        losses = 0
        size = 0

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []
        model_ascent.train()
        for idx, (data, target, is_poison, original_targets) in enumerate(
            train_loader, start=1
        ):
            data = self.args.tokenizer(
                data, padding=True, truncation=True, return_tensors="pt"
            )
            data["labels"] = target
            data = data.to(args.device)
            ret = self.model(**data)
            probs = F.softmax(ret.logits, dim=1)

            predictions = torch.argmax(probs, dim=1)

            loss_ascent = ret.loss
            losses += loss_ascent
            optimizer.zero_grad()
            loss_ascent.backward()
            optimizer.step()

            batch_loss_list.append(loss_ascent.item())
            batch_predict_list.append(torch.max(probs, -1)[1].detach().clone().cpu())
            batch_label_list.append(target.detach().clone().cpu())
            batch_poison_indicator_list.append(is_poison.detach().clone().cpu())
            batch_original_targets_list.append(original_targets.detach().clone().cpu())
        (
            train_epoch_loss_avg_over_batch,
            train_epoch_predict_list,
            train_epoch_label_list,
            train_epoch_poison_indicator_list,
            train_epoch_original_targets_list,
        ) = (
            sum(batch_loss_list) / len(batch_loss_list),
            torch.cat(batch_predict_list),
            torch.cat(batch_label_list),
            torch.cat(batch_poison_indicator_list),
            torch.cat(batch_original_targets_list),
        )

        train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
        train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
        train_clean_acc = all_acc(
            train_epoch_predict_list[train_clean_idx],
            train_epoch_label_list[train_clean_idx],
        )
        train_asr = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_label_list[train_bd_idx],
        )
        train_ra = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_original_targets_list[train_bd_idx],
        )
        return (
            train_epoch_loss_avg_over_batch,
            train_clean_acc,
            train_asr,
            train_ra,
        )

    def train_step_unlearn(
        self, args, train_loader, model_ascent, optimizer, criterion, epoch
    ):
        """Pretrain the model with raw data for each step
        args:
            Contains default parameters
        train_loader:
            the dataloader of train data
        model_ascent:
            the initial model
        optimizer:
            optimizer during the pretrain process
        criterion:
            criterion during the pretrain process
        epoch:
            current epoch
        """
        losses = 0
        size = 0

        batch_loss_list = []
        batch_predict_list = []
        batch_label_list = []
        batch_original_index_list = []
        batch_poison_indicator_list = []
        batch_original_targets_list = []

        model_ascent.train()

        for idx, (data, target, poison_indicator, original_targets) in enumerate(
            train_loader, start=1
        ):
            data = self.args.tokenizer(
                data, padding=True, truncation=True, return_tensors="pt"
            )
            data["labels"] = target
            data = data.to(args.device)
            ret = self.model(**data)
            probs = F.softmax(ret.logits, dim=1)

            predictions = torch.argmax(probs, dim=1)

            loss_ascent = ret.loss
            losses += loss_ascent

            optimizer.zero_grad()
            (-loss_ascent).backward()
            optimizer.step()

            batch_loss_list.append(loss_ascent.item())
            batch_predict_list.append(torch.max(probs, -1)[1].detach().clone().cpu())
            batch_label_list.append(target.detach().clone().cpu())
            batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
            batch_original_targets_list.append(original_targets.detach().clone().cpu())
        (
            train_epoch_loss_avg_over_batch,
            train_epoch_predict_list,
            train_epoch_label_list,
            train_epoch_poison_indicator_list,
            train_epoch_original_targets_list,
        ) = (
            sum(batch_loss_list) / len(batch_loss_list),
            torch.cat(batch_predict_list),
            torch.cat(batch_label_list),
            torch.cat(batch_poison_indicator_list),
            torch.cat(batch_original_targets_list),
        )

        train_bd_idx = torch.where(train_epoch_poison_indicator_list == 1)[0]
        train_clean_idx = torch.where(train_epoch_poison_indicator_list == 0)[0]
        train_clean_acc = all_acc(
            train_epoch_predict_list[train_clean_idx],
            train_epoch_label_list[train_clean_idx],
        )
        train_asr = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_label_list[train_bd_idx],
        )
        train_ra = all_acc(
            train_epoch_predict_list[train_bd_idx],
            train_epoch_original_targets_list[train_bd_idx],
        )

        return (
            train_epoch_loss_avg_over_batch,
            train_clean_acc,
            train_asr,
            train_ra,
        )

    def eval_step(
        self,
        netC,
        clean_test_dataloader,
        bd_test_dataloader,
        args,
    ):
        (
            clean_metrics,
            clean_epoch_predict_list,
            clean_epoch_label_list,
        ) = given_dataloader_test(
            args,
            netC,
            clean_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
        test_acc = clean_metrics["test_acc"]
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            args,
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
        test_asr = bd_metrics["test_acc"]
        ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
            args,
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        ra_test_loss_avg_over_batch = ra_metrics["test_loss_avg_over_batch"]
        test_ra = ra_metrics["test_acc"]
        return (
            clean_test_loss_avg_over_batch,
            bd_test_loss_avg_over_batch,
            ra_test_loss_avg_over_batch,
            test_acc,
            test_asr,
            test_ra,
        )
