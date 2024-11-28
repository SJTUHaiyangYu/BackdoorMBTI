'''
Neural Cleanse: Identifying And Mitigating Backdoor Attacks In Neural Networks
This file is modified based on the following source:
link : https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses

@inproceedings{wang2019neural,
    title={Neural cleanse: Identifying and mitigating backdoor attacks in neural networks},
    author={Wang, Bolun and Yao, Yuanshun and Shan, Shawn and Li, Huiying and Viswanath, Bimal and Zheng, Haitao and Zhao, Ben Y},
    booktitle={2019 IEEE Symposium on Security and Privacy (SP)},
    pages={707--723},
    year={2019},
    organization={IEEE}}
    
The following is the license from the original codebase
MIT License

Copyright (c) 2020 VinAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import sys

import cv2
import torch
from torch.utils.data import Subset

from defenses.base import DefenseBase
from utils.io import get_log_path_by_args

sys.path.append("../../")
sys.path.append(os.getcwd())
import random
from typing import Sequence
from train.sl_learning_train import SupervisedLearningTrain
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torchvision
from matplotlib import image as mlt
from torch.utils.data import DataLoader




class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (
                x[:, channel] - self.expected_values[channel]
            ) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (
                x[:, channel] * self.variance[channel] + self.expected_values[channel]
            )
        return x_clone


class RegressionModel(nn.Module):
    def __init__(self, opt, init_mask, init_pattern, model):
        self._EPSILON = opt.EPSILON
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))
        self.model = model
        self.classifier = self._get_classifier(opt)
        self.normalizer = self._get_normalize(opt)
        self.denormalizer = self._get_denormalize(opt)

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
            pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5

    def _get_classifier(self, opt):
        classifier = self.model
        classifier.to(opt.device)
        for param in classifier.parameters():
            param.requires_grad = False
        classifier.eval()
        return classifier.to(opt.device)

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10" or opt.dataset == "cifar100":
            denormalizer = Denormalize(
                opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
            )
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            denormalizer = None
        elif opt.dataset == "tiny":
            denormalizer = Denormalize(
                opt, [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
            )
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10" or opt.dataset == "cifar100":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            normalizer = None
        elif opt.dataset == "tiny":
            normalizer = Normalize(
                opt, [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
            )
        else:
            raise Exception("Invalid dataset")
        return normalizer


class Recorder:
    def __init__(self, opt):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = opt.init_cost
        self.cost_multiplier_up = opt.cost_multiplier
        self.cost_multiplier_down = opt.cost_multiplier**1.5

    def reset_state(self, opt):
        self.cost = opt.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

    def save_result_to_dir(self, opt):
        log_path = get_log_path_by_args(
            data_type=opt.data_type,
            attack_name=opt.attack_name,
            dataset=opt.dataset,
            model_name=opt.model_name,
            pratio=opt.pratio,
        )
        result_dir = log_path / "nc" / "results"
        result_dir = result_dir.as_posix()
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, str(opt.target_label))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        pattern_best = self.pattern_best
        mask_best = self.mask_best
        trigger = pattern_best * mask_best

        path_mask = os.path.join(result_dir, "mask.png")
        path_pattern = os.path.join(result_dir, "pattern.png")
        path_trigger = os.path.join(result_dir, "trigger.png")

        torchvision.utils.save_image(mask_best, path_mask, normalize=True)
        torchvision.utils.save_image(pattern_best, path_pattern, normalize=True)
        torchvision.utils.save_image(trigger, path_trigger, normalize=True)


def train_mask(args, model, trainloader, init_mask, init_pattern):
    # Build regression model
    regression_model = RegressionModel(args, init_mask, init_pattern, model).to(
        args.device
    )

    # Set optimizer
    optimizerR = torch.optim.Adam(
        regression_model.parameters(), lr=args.mask_lr, betas=(0.5, 0.9)
    )

    # Set recorder (for recording best result)
    recorder = Recorder(args)

    for epoch in range(args.nc_epoch):
        # early_stop = train_step(regression_model, optimizerR, test_dataloader, recorder, epoch, opt)
        early_stop = train_step(
            regression_model, optimizerR, trainloader, recorder, epoch, args
        )
        if early_stop:
            break

    # Save result to dir
    recorder.save_result_to_dir(args)

    return recorder, args


def train_step(regression_model, optimizerR, dataloader, recorder, epoch, opt):
    # print("Epoch {} - Label: {} | {} - {}:".format(epoch, opt.target_label, opt.dataset, opt.attack_mode))
    # Set losses
    cross_entropy = nn.CrossEntropyLoss()
    total_pred = 0
    true_pred = 0

    # Record loss for all mini-batches
    loss_ce_list = []
    loss_reg_list = []
    loss_list = []
    loss_acc_list = []

    # Set inner early stop flag
    inner_early_stop_flag = False
    for batch_idx, (inputs, labels, *other_info) in enumerate(dataloader):
        # Forwarding and update model
        optimizerR.zero_grad()

        inputs = inputs.to(opt.device)
        sample_num = inputs.shape[0]
        total_pred += sample_num
        target_labels = (
            torch.ones((sample_num), dtype=torch.int64).to(opt.device)
            * opt.target_label
        )
        predictions = regression_model(inputs)

        loss_ce = cross_entropy(predictions, target_labels)
        loss_reg = torch.norm(regression_model.get_raw_mask(), opt.use_norm)
        total_loss = loss_ce + recorder.cost * loss_reg
        total_loss.backward()
        optimizerR.step()

        # Record minibatch information to list
        minibatch_accuracy = (
            torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
            * 100.0
            / sample_num
        )
        loss_ce_list.append(loss_ce.detach())
        loss_reg_list.append(loss_reg.detach())
        loss_list.append(total_loss.detach())
        loss_acc_list.append(minibatch_accuracy)

        true_pred += torch.sum(
            torch.argmax(predictions, dim=1) == target_labels
        ).detach()
        # progress_bar(batch_idx, len(dataloader))

    loss_ce_list = torch.stack(loss_ce_list)
    loss_reg_list = torch.stack(loss_reg_list)
    loss_list = torch.stack(loss_list)
    loss_acc_list = torch.stack(loss_acc_list)

    avg_loss_ce = torch.mean(loss_ce_list)
    avg_loss_reg = torch.mean(loss_reg_list)
    avg_loss = torch.mean(loss_list)
    avg_loss_acc = torch.mean(loss_acc_list)

    # Check to save best mask or not
    if avg_loss_acc >= opt.atk_succ_threshold and avg_loss_reg < recorder.reg_best:
        recorder.mask_best = regression_model.get_raw_mask().detach()
        recorder.pattern_best = regression_model.get_raw_pattern().detach()
        recorder.reg_best = avg_loss_reg
        recorder.save_result_to_dir(opt)
        # print(" Updated !!!")

    # Show information
    # print(
    #     "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
    #         true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
    #     )
    # )

    # Check early stop
    if opt.early_stop:
        if recorder.reg_best < float("inf"):
            if (
                recorder.reg_best
                >= opt.early_stop_threshold * recorder.early_stop_reg_best
            ):
                recorder.early_stop_counter += 1
            else:
                recorder.early_stop_counter = 0

        recorder.early_stop_reg_best = min(
            recorder.early_stop_reg_best, recorder.reg_best
        )

        if (
            recorder.cost_down_flag
            and recorder.cost_up_flag
            and recorder.early_stop_counter >= opt.early_stop_patience
        ):

            inner_early_stop_flag = True

    if not inner_early_stop_flag:
        # Check cost modification
        if recorder.cost == 0 and avg_loss_acc >= opt.atk_succ_threshold:
            recorder.cost_set_counter += 1
            if recorder.cost_set_counter >= opt.patience:
                recorder.reset_state(opt)
        else:
            recorder.cost_set_counter = 0

        if avg_loss_acc >= opt.atk_succ_threshold:
            recorder.cost_up_counter += 1
            recorder.cost_down_counter = 0
        else:
            recorder.cost_up_counter = 0
            recorder.cost_down_counter += 1

        if recorder.cost_up_counter >= opt.patience:
            recorder.cost_up_counter = 0
            print(
                "Up cost from {} to {}".format(
                    recorder.cost, recorder.cost * recorder.cost_multiplier_up
                )
            )
            recorder.cost *= recorder.cost_multiplier_up
            recorder.cost_up_flag = True

        elif recorder.cost_down_counter >= opt.patience:
            recorder.cost_down_counter = 0
            print(
                "Down cost from {} to {}".format(
                    recorder.cost, recorder.cost / recorder.cost_multiplier_down
                )
            )
            recorder.cost /= recorder.cost_multiplier_down
            recorder.cost_down_flag = True

        # Save the final version
        if recorder.mask_best is None:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()

    del predictions
    torch.cuda.empty_cache()

    return inner_early_stop_flag


def outlier_detection(l1_norm_list, idx_mapping, opt):
    print("-" * 30)
    print("Determining whether model is backdoor")
    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print("Median: {}, MAD: {}".format(median, mad))
    print("Anomaly index: {}".format(min_mad))

    if min_mad < 2:
        print("Not a backdoor model")
    else:
        print("This is a backdoor model")

    if opt.to_file:
        # result_path = os.path.join(opt.result, opt.saving_prefix, opt.dataset)
        # output_path = os.path.join(
        #     result_path, "{}_{}_output.txt".format(opt.attack_mode, opt.dataset, opt.attack_mode)
        # )
        output_path = opt.output_path
        with open(output_path, "a+") as f:
            f.write(
                str(median.cpu().numpy())
                + ", "
                + str(mad.cpu().numpy())
                + ", "
                + str(min_mad.cpu().numpy())
                + "\n"
            )
            l1_norm_list_to_save = [str(value) for value in l1_norm_list.cpu().numpy()]
            f.write(", ".join(l1_norm_list_to_save) + "\n")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])
    print(
        "Flagged label list: {}".format(
            ",".join(
                ["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]
            )
        )
    )
    print(flag_list)
    return flag_list


class Nc(DefenseBase):
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
            clean_train_set,
            clean_test_set,
            poison_train_set,
            poison_test_set,
            model,
            collate_fn,
        )
        self.model = model
        self.ImageWappermodel = model

    def train(self):
        # prepare optimizer, scheduler, critertion
        # a. initialize the model and trigger
        result_path = os.getcwd() + "/" + f"{self.args.save_folder_name}/trigger/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.args.output_path = result_path + "{}_output_clean.txt".format(
            self.args.dataset
        )
        with open(self.args.output_path, "w+") as f:
            f.write("Output for cleanse:  - {}".format(self.args.dataset) + "\n")
        init_mask = np.ones((1, self.args.input_height, self.args.input_width)).astype(
            np.float32
        )
        init_pattern = np.ones(
            (self.args.input_channel, self.args.input_height, self.args.input_width)
        ).astype(np.float32)
        flag = 0
        for test in range(self.args.n_times_test):
            # b. train triggers according to different target labels
            print("Test {}:".format(test))
            masks = []
            idx_mapping = {}

            for target_label in range(self.args.num_classes):
                print(
                    "----------------- Analyzing label: {} -----------------".format(
                        target_label
                    )
                )
                self.args.target_label = target_label
                recorder, self.args = train_mask(
                    self.args,
                    self.model,
                    self.clean_train_loader,
                    init_mask,
                    init_pattern,
                )
                mask = recorder.mask_best
                masks.append(mask)
                reg = torch.norm(mask, p=self.args.use_norm)
                print(
                    f"The regularization of mask for target label {target_label} is {reg}"
                )
                idx_mapping[target_label] = len(masks) - 1
            # c. Determine whether the trained reverse trigger is a real backdoor trigger

            l1_norm_list = torch.stack(
                [torch.norm(m, p=self.args.use_norm) for m in masks]
            )
            print("{} labels found".format(len(l1_norm_list)))
            print("Norm values: {}".format(l1_norm_list))
            flag_list = outlier_detection(l1_norm_list, idx_mapping, self.args)
            for param in self.model.to(self.args.device).parameters():
                param.requires_grad = True
            # flag_list = [(0, tensor(5.1644, device='cuda:0'))]
            if len(flag_list) != 0:
                flag = 1
        if flag == 0:
            # without triggers,only test and return
            print("This is not a backdoor model")
            test_dataloader_dict = {}
            test_dataloader_dict["clean_test_dataloader"] = self.clean_test_loader
            test_dataloader_dict["bd_test_dataloader"] = self.poison_test_loader

            results = super().eval()
            return results
        # d. select samples as clean samples and unlearning samples, finetune the origin model

        # self.model
        clean_dataset = self.clean_train_loader.dataset
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length)
        log_index = "index.txt"
        np.savetxt(log_index, ran_idx, fmt="%d")
        clean_dataset = Subset(clean_dataset, ran_idx)
        # clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = clean_dataset
        data_set_o.wrapped_dataset = data_set_without_tran
        x_new = list()
        y_new = list()
        original_index_array = list()
        poison_indicator = list()
        for ii in range(int(len(data_set_o) * (1 - self.args.unlearning_ratio))):
            x_new.extend([data_set_o.wrapped_dataset[ii][0]])
            y_new.extend([data_set_o.wrapped_dataset[ii][1]])
            original_index_array.extend([len(x_new) - 1])
            poison_indicator.extend([0])
        log_path = get_log_path_by_args(
            data_type=self.args.data_type,
            attack_name=self.args.attack_name,
            dataset=self.args.dataset,
            model=self.args.model,
            pratio=self.args.pratio,
        )
        result_dir = log_path / "nc" / "results"
        for label, _ in flag_list:
            mask_path = str(result_dir) + "/{}".format(str(label)) + "/mask.png"
            mask_image = mlt.imread(mask_path)
            mask_image = cv2.resize(
                mask_image, (self.args.input_height, self.args.input_width)
            )
            trigger_path = str(result_dir) + "/{}".format(str(label)) + "/trigger.png"
            signal_mask = mlt.imread(trigger_path) * 255
            signal_mask = cv2.resize(
                signal_mask, (self.args.input_height, self.args.input_width)
            )

            x_unlearn = list()
            x_unlearn_new = list()
            y_unlearn_new = list()
            original_index_array_new = list()
            poison_indicator_new = list()
            for ii in range(
                int(len(data_set_o) * (1 - self.args.unlearning_ratio)),
                int(len(data_set_o)),
            ):
                img = data_set_o.wrapped_dataset[ii][0]
                img = np.transpose(img.numpy(), (1, 2, 0))
                x_unlearn.extend([img])
                x_np = np.array(
                    cv2.resize(
                        np.array(img), (self.args.input_height, self.args.input_width)
                    )
                ) * (1 - np.array(mask_image)) + np.array(signal_mask)
                x_np = np.clip(x_np.astype("uint8"), 0, 255)
                x_tensor = torch.from_numpy(np.transpose(x_np, (2, 0, 1)))
                x_unlearn_new.extend([x_tensor])
                y_unlearn_new.extend([data_set_o.wrapped_dataset[ii][1]])
                original_index_array_new.extend([len(x_new) - 1])
                poison_indicator_new.extend([0])
            x_new.extend(x_unlearn_new)
            y_new.extend(y_unlearn_new)
            original_index_array.extend(original_index_array_new)
            poison_indicator.extend(poison_indicator_new)
        ori_dataset = xy_iter(x_new, y_new, None)

        trainloader = DataLoader(
            ori_dataset,
            batch_size=self.args.batch_size,
            drop_last=False,
            shuffle=True,
            collate_fn=None,
        )
        train_loader = self.fabric.setup_dataloaders(trainloader)
        train =  SupervisedLearningTrain(train_loader, self.args)
        train.train_model()
        results = super().eval()

        return results


def choose_index(args, data_all_length):
    # choose clean data according to index

    ran_idx = random.sample(range(data_all_length), int(data_all_length * args.ratio))

    return ran_idx


class xy_iter(torch.utils.data.dataset.Dataset):
    def __init__(self, x: Sequence, y: Sequence, transform):
        assert len(x) == len(y)
        super().__init__()
        self.data = x
        self.targets = y
        self.transform = transform

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, 0, label

    def __len__(self):
        return len(self.targets)
