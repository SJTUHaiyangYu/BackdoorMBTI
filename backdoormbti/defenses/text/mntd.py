import os
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from defenses.base import DefenseBase


def load_dataset_setting(task):
    if task == "mnist":
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = torchvision.datasets.MNIST(
            root="./raw_data/", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root="./raw_data/", train=False, download=False, transform=transform
        )
        is_binary = False
        need_pad = False
        from model_lib.mnist_cnn_model import Model, troj_gen_func, random_troj_setting
    elif task == "cifar10":
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = torchvision.datasets.CIFAR10(
            root="./raw_data/", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./raw_data/", train=False, download=False, transform=transform
        )
        is_binary = False
        need_pad = False
        from model_lib.cifar10_cnn_model import (
            Model,
            troj_gen_func,
            random_troj_setting,
        )
    elif task == "audio":
        BATCH_SIZE = 100
        N_EPOCH = 100
        from model_lib.audio_dataset import SpeechCommand

        trainset = SpeechCommand(split=0)
        testset = SpeechCommand(split=2)
        is_binary = False
        need_pad = False
        from model_lib.audio_rnn_model import Model, troj_gen_func, random_troj_setting
    elif task == "rtNLP":
        BATCH_SIZE = 64
        N_EPOCH = 50
        from model_lib.rtNLP_dataset import RTNLP

        trainset = RTNLP(train=True)
        testset = RTNLP(train=False)
        is_binary = True
        need_pad = True
        from model_lib.rtNLP_cnn_model import Model, troj_gen_func, random_troj_setting
    else:
        raise NotImplementedError("Unknown task %s" % task)

    return (
        BATCH_SIZE,
        N_EPOCH,
        trainset,
        testset,
        is_binary,
        need_pad,
        Model,
        troj_gen_func,
        random_troj_setting,
    )


class BackdoorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        src_dataset,
        atk_setting,
        troj_gen_func,
        choice=None,
        mal_only=False,
        need_pad=False,
    ):
        self.src_dataset = src_dataset
        self.atk_setting = atk_setting
        self.troj_gen_func = troj_gen_func
        self.need_pad = need_pad

        self.mal_only = mal_only
        if choice is None:
            choice = np.arange(len(src_dataset))
        self.choice = choice
        inject_p = atk_setting[5]
        self.mal_choice = np.random.choice(
            choice, int(len(choice) * inject_p), replace=False
        )

    def __len__(
        self,
    ):
        if self.mal_only:
            return len(self.mal_choice)
        else:
            return len(self.choice) + len(self.mal_choice)

    def __getitem__(self, idx):
        if not self.mal_only and idx < len(self.choice):
            # Return non-trojaned data
            if self.need_pad:
                # In NLP task we need to pad input with length of Troj pattern
                p_size = self.atk_setting[0]
                X, y = self.src_dataset[self.choice[idx]]
                X_padded = torch.cat([X, torch.LongTensor([0] * p_size)], dim=0)
                return X_padded, y
            else:
                return self.src_dataset[self.choice[idx]]

        if self.mal_only:
            X, y = self.src_dataset[self.mal_choice[idx]]
        else:
            X, y = self.src_dataset[self.mal_choice[idx - len(self.choice)]]
        X_new, y_new = self.troj_gen_func(X, y, self.atk_setting)
        return X_new, y_new


def train_model(model, dataloader, epoch_num, is_binary, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i, (x_in, y_in) in enumerate(dataloader):
            B = x_in.size()[0]
            pred = model(x_in)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item() * B
            if is_binary:
                cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
        if verbose:
            print(
                "Epoch %d, loss = %.4f, acc = %.4f"
                % (epoch, cum_loss / tot, cum_acc / tot)
            )
    return


def eval_model(model, dataloader, is_binary):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        pred = model(x_in)
        if is_binary:
            cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot


class MNTDDefense(DefenseBase):
    def __init__(self, args) -> None:
        self.args = args
        assert args.troj_type in ("M", "B"), "unknown trojan pattern"

    def train(self):
        np.random.seed(0)
        torch.manual_seed(0)
        if args.GPU:
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        (
            BATCH_SIZE,
            N_EPOCH,
            trainset,
            testset,
            is_binary,
            need_pad,
            Model,
            troj_gen_func,
            random_troj_setting,
        ) = load_dataset_setting(args.task)
        tot_num = len(trainset)
        shadow_indices = np.random.choice(tot_num, int(tot_num * args.SHADOW_PROP))
        target_indices = np.random.choice(tot_num, int(tot_num * args.TARGET_PROP))
        print("Data indices owned by the attacker:", target_indices)

        SAVE_PREFIX = "./shadow_model_ckpt/%s" % args.dataset
        if not os.path.isdir(SAVE_PREFIX):
            os.mkdir(SAVE_PREFIX)
        if not os.path.isdir(SAVE_PREFIX + "/models"):
            os.mkdir(SAVE_PREFIX + "/models")

        all_target_acc = []
        all_target_acc_mal = []

        for i in range(args.TARGET_NUM):
            model = Model(gpu=args.GPU)
            atk_setting = random_troj_setting(args.troj_type)
            trainset_mal = BackdoorDataset(
                trainset,
                atk_setting,
                troj_gen_func,
                choice=target_indices,
                need_pad=need_pad,
            )
            trainloader = torch.utils.data.DataLoader(
                trainset_mal, batch_size=BATCH_SIZE, shuffle=True
            )
            testset_mal = BackdoorDataset(
                testset, atk_setting, troj_gen_func, mal_only=True
            )
            testloader_benign = torch.utils.data.DataLoader(
                testset, batch_size=BATCH_SIZE
            )
            testloader_mal = torch.utils.data.DataLoader(
                testset_mal, batch_size=BATCH_SIZE
            )

            train_model(
                model,
                trainloader,
                epoch_num=int(N_EPOCH * args.SHADOW_PROP / args.TARGET_PROP),
                is_binary=is_binary,
                verbose=False,
            )
            save_path = SAVE_PREFIX + "/models/target_troj%s_%d.model" % (
                args.troj_type,
                i,
            )
            torch.save(model.state_dict(), save_path)
            acc = eval_model(model, testloader_benign, is_binary=is_binary)
            acc_mal = eval_model(model, testloader_mal, is_binary=is_binary)
            print(
                "Acc %.4f, Acc on backdoor %.4f, saved to %s @ %s"
                % (acc, acc_mal, save_path, datetime.now())
            )
            p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
            print(
                "\tp size: %d; loc: %s; alpha: %.3f; target_y: %d; inject p: %.3f"
                % (p_size, loc, alpha, target_y, inject_p)
            )
            all_target_acc.append(acc)
            all_target_acc_mal.append(acc_mal)

        log = {
            "target_num": args.TARGET_NUM,
            "target_acc": sum(all_target_acc) / len(all_target_acc),
            "target_acc_mal": sum(all_target_acc_mal) / len(all_target_acc_mal),
        }
        log_path = SAVE_PREFIX + "/troj%s.log" % args.troj_type
        with open(log_path, "w") as outf:
            json.dump(log, outf)
        print("Log file saved to %s" % log_path)
