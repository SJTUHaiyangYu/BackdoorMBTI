"""
Anti-backdoor learning: Training clean models on poisoned data.
this file is for abl defense
link : https://github.com/bboylyg/ABL.

@article{li2021anti,
            title={Anti-backdoor learning: Training clean models on poisoned data},
            author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            pages={14900--14912},
            year={2021}
            }
"""

from abc import abstractmethod
import os
import sys


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset
from tqdm import tqdm

from defenses.base import DefenseBase
from utils.collate_fn import AudioCollator

from utils.model import load_model


sys.path.append("../")
sys.path.append(os.getcwd())

import logging


class LGALoss(nn.Module):
    def __init__(self, gamma, criterion):
        super(LGALoss, self).__init__()
        self.gamma = gamma
        self.criterion = criterion

    def forward(self, output, target):
        loss = self.criterion(output, target)
        # add Local Gradient Ascent(LGA) loss
        loss_ascent = torch.sign(loss - self.gamma) * loss
        return loss_ascent


class FloodingLoss(nn.Module):
    def __init__(self, flooding, criterion):
        super(FloodingLoss, self).__init__()
        self.flooding = flooding
        self.criterion = criterion

    def forward(self, output, target):
        loss = self.criterion(output, target)
        # add Local Gradient Ascent(LGA) loss
        loss_ascent = (loss - self.flooding).abs() + self.flooding
        return loss_ascent


def adjust_learning_rate(optimizer, epoch, args):
    """set learning rate during the process of pretraining model
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    """
    if epoch < args.tuning_epochs:
        lr = args.lr
    else:
        lr = 0.01
    logging.info("epoch: {}  lr: {:.4f}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def compute_loss_value(args, poisoned_data, model_ascent):
    """Calculate loss value per example
    args:
        Contains default parameters
    poisoned_data:
        the train dataset which contains backdoor data
    model_ascent:
        the model after the process of pretrain
    """
    # Define loss function
    if args.device == "cuda":
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_ascent.eval()
    losses_record = []
    if args.data_type == "audio":
        collate_fn = AudioCollator(args)
    else:
        collate_fn = None
    example_data_loader = torch.utils.data.DataLoader(
        dataset=poisoned_data, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    if args.data_type == "text":
        for idx, (data, target, _, _) in tqdm(enumerate(example_data_loader, start=0)):
            data = args.tokenizer(
                data, padding=True, truncation=True, return_tensors="pt"
            )
            data["labels"] = target
            data = data.to(args.device)

            with torch.no_grad():
                ret = model_ascent(**data)
                loss = ret.loss
            losses_record.append(loss.item())
    elif args.data_type == "image":
        for idx, (img, target, _, _) in tqdm(enumerate(example_data_loader, start=0)):
            img = img.to(args.device)
            target = target.to(args.device)

            with torch.no_grad():
                output = model_ascent(img)
                loss = criterion(output, target)
            losses_record.append(loss.item())
    elif args.data_type == "audio":
        if args.dataset == "speechcommands":
            for idx, (data, target, _, _) in tqdm(
                enumerate(example_data_loader, start=0)
            ):
                if data.shape != torch.Size([1, 1, 16000]):
                    continue
                data = data.to(args.device)
                target = target.to(args.device)
                with torch.no_grad():
                    pred = model_ascent(data)
                    pred = pred.squeeze(dim=1)
                    loss = criterion(pred, target)

                losses_record.append(loss.item())
        elif args.dataset == "gtzan" or args.dataset == "voxceleb1idenfication":
            for idx, (data, target, _, _) in tqdm(
                enumerate(example_data_loader, start=0)
            ):
                # if data.shape != torch.Size([1, 1, size]):
                #     continue
                data = data.to(args.device)
                target = target.to(args.device)
                with torch.no_grad():
                    pred = model_ascent(data)
                    pred = pred.squeeze(dim=1)
                    loss = criterion(pred, target)

                losses_record.append(loss.item())

    losses_idx = np.argsort(
        np.array(losses_record)
    )  # get the index of examples by loss value in descending order

    # Show the top 10 loss values
    losses_record_arr = np.array(losses_record)
    logging.info(f"Top ten loss value: {losses_record_arr[losses_idx[:10]]}")
    print(f"Top ten loss value: {losses_record_arr[losses_idx[:10]]}")
    return losses_idx


def isolate_data(args, poison_data, losses_idx):
    """isolate the backdoor data with the calculated loss
    args:
        Contains default parameters
    result:
        the attack result contain the train dataset which contains backdoor data
    losses_idx:
        the index of order about the loss value for each data
    """
    # Initialize lists
    ratio = args.isolation_ratio
    perm = losses_idx[0 : int(len(losses_idx) * ratio)]
    permnot = losses_idx[int(len(losses_idx) * ratio) :]
    train_dataset = poison_data
    data_set_isolate = poison_data
    data_set_other = train_dataset
    perm = [int(i) for i in perm]
    permnot = [int(i) for i in permnot]
    data_set_isolate = Subset(data_set_isolate, perm)
    data_set_other = Subset(data_set_other, permnot)
    print("Finish collecting {} isolation examples: ".format(len(data_set_isolate)))
    print("Finish collecting {} other examples: ".format(len(data_set_other)))
    logging.info(
        "Finish collecting {} isolation examples: ".format(len(data_set_isolate))
    )
    logging.info("Finish collecting {} other examples: ".format(len(data_set_other)))

    return data_set_isolate, data_set_other


def learning_rate_finetuning(optimizer, epoch, args):
    """set learning rate during the process of finetuing model
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    """
    if epoch < 40:
        lr = 0.01
    elif epoch < 60:
        lr = 0.001
    else:
        lr = 0.001
    logging.info("epoch: {}  lr: {:.4f}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def learning_rate_unlearning(optimizer, epoch, args):
    """set learning rate during the process of unlearning model
    optimizer:
        optimizer during the pretrain process
    epoch:
        current epoch
    args:
        Contains default parameters
    """
    if epoch < args.unlearning_epochs:
        lr = 0.0001
    else:
        lr = 0.0001
    logging.info("epoch: {}  lr: {:.4f}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
        if args.data_type == "text":
            for batch_idx, (data, target, *additional_info) in enumerate(
                test_dataloader
            ):
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
        elif args.data_type == "image":
            for batch_idx, (x, target, *additional_info) in enumerate(test_dataloader):
                x = x.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                if verbose == 1:
                    batch_predict_list.append(predicted.detach().clone().cpu())
                    batch_label_list.append(target.detach().clone().cpu())

                metrics["test_correct"] += correct.item()
                metrics["test_loss_sum_over_batch"] += loss.item()
                metrics["test_total"] += target.size(0)
        elif args.data_type == "audio":
            for batch_idx, (x, target, *additional_info) in enumerate(test_dataloader):
                x = x.to(device, non_blocking=non_blocking)
                target = target.to(device, non_blocking=non_blocking)
                pred = model(x)
                pred = pred.squeeze()
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
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


class Abl_Base(DefenseBase):
    def __init__(self, args) -> None:
        super().__init__(args=args)

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
        ###a. pre-train model
        poisoned_data, model_ascent = self.pre_train(
            self.args,
            self.poison_train_loader,
            self.poison_test_loader,
            self.clean_train_loader,
            self.clean_test_loader,
        )
        ###b. isolate the special data(loss is low) as backdoor data
        losses_idx = compute_loss_value(self.args, poisoned_data, model_ascent)
        logging.info("----------- Collect isolation data -----------")
        isolation_examples, other_examples = isolate_data(
            self.args, poisoned_data, losses_idx
        )
        ###c. unlearn the backdoor data and learn the remaining data
        model_new = self.train_unlearning(
            self.args,
            self.clean_test_loader,
            self.poison_test_loader,
            model_ascent,
            isolation_examples,
            other_examples,
        )
        self.args.model = model_new
        results = super().eval()
        return results

    def pre_train(
        self,
        args,
        poison_train_loader,
        poison_test_loader,
        clean_train_loader,
        clean_test_loader,
    ):
        """Pretrain the model with raw data
        args:
            Contains default parameters
        result:
            attack result(details can be found in utils)
        """
        # Load models
        logging.info("----------- Network Initialization --------------")
        # self.args.model = self.args.model_name
        model_ascent = load_model(self.args)

        logging.info("finished model init...")
        # initialize optimizer
        # because the optimizer has parameter nesterov
        optimizer = torch.optim.SGD(
            model_ascent.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

        # define loss functions
        # recommend to use cross entropy
        criterion = nn.CrossEntropyLoss().to(self.args.device)
        if args.gradient_ascent_type == "LGA":
            criterion = LGALoss(args.gamma, criterion).to(args.device)
        elif args.gradient_ascent_type == "Flooding":
            criterion = FloodingLoss(args.flooding, criterion).to(args.device)
        else:
            raise NotImplementedError
        logging.info("----------- Data Initialization --------------")
        logging.info("----------- Train Initialization --------------")
        for epoch in range(0, args.tuning_epochs):
            logging.info("Epoch {}:".format(epoch + 1))
            adjust_learning_rate(optimizer, epoch, args)
            (
                train_epoch_loss_avg_over_batch,
                train_clean_acc,
                train_asr,
                train_ra,
            ) = self.train_step(
                args, poison_train_loader, model_ascent, optimizer, criterion, epoch + 1
            )
            print(f"train_clean_acc:{train_clean_acc}")
            print(f"train_asr:{train_asr}")
            (
                clean_test_loss_avg_over_batch,
                bd_test_loss_avg_over_batch,
                ra_test_loss_avg_over_batch,
                test_acc,
                test_asr,
                test_ra,
            ) = self.eval_step(
                model_ascent,
                clean_test_loader,
                poison_test_loader,
                args,
            )
            print(f"test_acc:{test_acc}")
            print(f"test_asr:{test_asr}")
        return poison_train_loader.dataset, model_ascent

    def train_unlearning(
        self,
        args,
        clean_test_loader,
        poison_test_loader,
        model_ascent,
        isolate_poisoned_data,
        isolate_other_data,
    ):
        """train the model with remaining data and unlearn the backdoor data
        args:
            Contains default parameters
        result:
            attack result(details can be found in utils)
        model_ascent:
            the model after pretrain
        isolate_poisoned_data:
            the dataset of 'backdoor' data
        isolate_other_data:
            the dataset of remaining data
        """
        logging.info("Finish loading ascent model...")
        # initialize optimizer
        # Because nesterov we do not use other optimizer
        optimizer = torch.optim.SGD(
            model_ascent.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

        # define loss functions
        # you can use other criterion, but the paper use cross validation to unlearn sample
        criterion = nn.CrossEntropyLoss().to(self.args.device)
        isolate_poisoned_data_loader = torch.utils.data.DataLoader(
            dataset=isolate_poisoned_data,
            batch_size=args.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

        isolate_other_data_loader = torch.utils.data.DataLoader(
            dataset=isolate_other_data,
            batch_size=args.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

        data_bd_testset = poison_test_loader.dataset
        data_bd_loader = torch.utils.data.DataLoader(
            data_bd_testset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=args.pin_memory,
        )

        data_clean_testset = clean_test_loader.dataset
        data_clean_loader = torch.utils.data.DataLoader(
            data_clean_testset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=args.pin_memory,
        )

        train_loss_list = []
        train_clean_acc_list = []
        train_asr_list = []
        train_ra_list = []

        clean_test_loss_list = []
        bd_test_loss_list = []
        ra_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        logging.info("----------- Train Initialization --------------")

        if args.finetuning_ascent_model == True:
            # this is to improve the clean accuracy of isolation model, you can skip this step
            logging.info("----------- Finetuning isolation model --------------")
            for epoch in range(0, args.finetuning_epochs):
                learning_rate_finetuning(optimizer, epoch, args)
                (
                    train_epoch_loss_avg_over_batch,
                    train_clean_acc,
                    train_asr,
                    train_ra,
                ) = self.train_step(
                    args,
                    isolate_other_data_loader,
                    model_ascent,
                    optimizer,
                    criterion,
                    epoch + 1,
                )

                (
                    clean_test_loss_avg_over_batch,
                    bd_test_loss_avg_over_batch,
                    ra_test_loss_avg_over_batch,
                    test_acc,
                    test_asr,
                    test_ra,
                ) = self.eval_step(
                    model_ascent,
                    data_clean_loader,
                    data_bd_loader,
                    args,
                )

                train_loss_list.append(train_epoch_loss_avg_over_batch)
                train_clean_acc_list.append(train_clean_acc)
                train_asr_list.append(train_asr)
                train_ra_list.append(train_ra)

                clean_test_loss_list.append(clean_test_loss_avg_over_batch)
                bd_test_loss_list.append(bd_test_loss_avg_over_batch)
                ra_test_loss_list.append(ra_test_loss_avg_over_batch)
                test_acc_list.append(test_acc)
                test_asr_list.append(test_asr)
                test_ra_list.append(test_ra)

        logging.info("----------- Model unlearning --------------")
        for epoch in range(0, args.unlearning_epochs):
            learning_rate_unlearning(optimizer, epoch, args)
            (
                train_epoch_loss_avg_over_batch,
                train_clean_acc,
                train_asr,
                train_ra,
            ) = self.train_step_unlearn(
                args,
                isolate_poisoned_data_loader,
                model_ascent,
                optimizer,
                criterion,
                epoch + 1,
            )

            (
                clean_test_loss_avg_over_batch,
                bd_test_loss_avg_over_batch,
                ra_test_loss_avg_over_batch,
                test_acc,
                test_asr,
                test_ra,
            ) = self.eval_step(
                model_ascent,
                data_clean_loader,
                data_bd_loader,
                args,
            )

            train_loss_list.append(train_epoch_loss_avg_over_batch)
            train_clean_acc_list.append(train_clean_acc)
            train_asr_list.append(train_asr)
            train_ra_list.append(train_ra)

            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            ra_test_loss_list.append(ra_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)
        print(f"test_acc_list:{test_acc_list}")
        print(f"test_asr_list:{test_asr_list}")
        return model_ascent

    @abstractmethod
    def train_step(
        self, args, train_loader, model_ascent, optimizer, criterion, epoch
    ): ...

    @abstractmethod
    def train_step_unlearn(
        self, args, train_loader, model_ascent, optimizer, criterion, epoch
    ): ...

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
