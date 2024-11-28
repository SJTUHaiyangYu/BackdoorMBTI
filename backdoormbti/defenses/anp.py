import copy
import logging

import numpy as np
import torch
from torch import nn
from defenses.base import DefenseBase


from torch import Tensor
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import time
import os
import pandas as pd
from copy import deepcopy


def clip_mask(model, lower=0.0, upper=1.0):
    params = [
        param for name, param in model.named_parameters() if "neuron_mask" in name
    ]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [
        (name, param)
        for name, param in model.named_parameters()
        if "neuron_noise" in name
    ]
    for name, p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init, args):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


def mask_train(model, criterion, mask_opt, noise_opt, data_loader, args):
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, sample in enumerate(data_loader):
        images, labels, _, _ = sample
        images, labels = images.to(args.device), labels.to(args.device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(model, rand_init=True, args=args)
            for _ in range(args.anp_steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = -criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if args.anp_eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def test(model, criterion, data_loader, args):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            images, labels, _, _ = sample
            images, labels = images.to(args.device), labels.to(args.device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if "neuron_mask" in name:
            for idx in range(param.size(0)):
                neuron_name = ".".join(name.split(".")[:-1])
                mask_values.append(
                    "{} \t {} \t {} \t {:.4f} \n".format(
                        count, neuron_name, idx, param[idx].item()
                    )
                )
                count += 1
    with open(file_name, "w") as f:
        f.write("No \t Layer Name \t Neuron Idx \t Mask Score \n")
        f.writelines(mask_values)


def read_data(file_name):
    tempt = pd.read_csv(file_name, sep="\s+", skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = "{}.{}".format(neuron[0], "weight")
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


def evaluate_by_number(
    model,
    mask_values,
    pruning_max,
    pruning_step,
    criterion,
    clean_loader,
    poison_loader,
    args,
):
    results = []
    nb_max = int(np.ceil(pruning_max))
    print(nb_max)
    nb_step = int(np.ceil(pruning_step))
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = (
            mask_values[i][0],
            mask_values[i][1],
            mask_values[i][2],
        )
        cl_loss, cl_acc = test(
            model=model, criterion=criterion, data_loader=clean_loader, args=args
        )
        po_loss, po_acc = test(
            model=model, criterion=criterion, data_loader=poison_loader, args=args
        )
        print(
            "{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
                i + 1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc
            )
        )
        results.append(
            "{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
                i + 1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc
            )
        )
    return results


class NoisyBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(NoisyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.neuron_mask = Parameter(torch.Tensor(num_features), requires_grad=True)
        self.neuron_noise = Parameter(torch.Tensor(num_features), requires_grad=True)
        self.neuron_noise_bias = Parameter(
            torch.Tensor(num_features), requires_grad=True
        )
        init.ones_(self.neuron_mask)
        init.zeros_(self.neuron_noise)
        init.zeros_(self.neuron_noise_bias)
        self.is_perturbed = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.neuron_noise, a=-eps, b=eps)
            init.uniform_(self.neuron_noise_bias, a=-eps, b=eps)
        else:
            init.zeros_(self.neuron_noise)
            init.zeros_(self.neuron_noise_bias)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        if self.is_perturbed:
            coeff_weight = self.neuron_mask + self.neuron_noise
            coeff_bias = 1.0 + self.neuron_noise_bias
        else:
            coeff_weight = self.neuron_mask
            coeff_bias = 1.0
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            (
                self.running_mean
                if not self.training or self.track_running_stats
                else None
            ),
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight,
            self.bias * coeff_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class NoisyBatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(NoisyBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.neuron_mask_fc = Parameter(torch.Tensor(num_features), requires_grad=True)
        self.neuron_noise_fc = Parameter(torch.Tensor(num_features), requires_grad=True)
        self.neuron_noise_bias_fc = Parameter(
            torch.Tensor(num_features), requires_grad=True
        )
        init.ones_(self.neuron_mask_fc)
        init.zeros_(self.neuron_noise_fc)
        init.zeros_(self.neuron_noise_bias_fc)
        self.is_perturbed = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.neuron_noise_fc, a=-eps, b=eps)
            init.uniform_(self.neuron_noise_bias_fc, a=-eps, b=eps)
        else:
            init.zeros_(self.neuron_noise_fc)
            init.zeros_(self.neuron_noise_bias_fc)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        if self.is_perturbed:
            coeff_weight = self.neuron_mask_fc + self.neuron_noise_fc
            coeff_bias = 1.0 + self.neuron_noise_bias_fc
        else:
            coeff_weight = self.neuron_mask_fc
            coeff_bias = 1.0
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            (
                self.running_mean
                if not self.training or self.track_running_stats
                else None
            ),
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight,
            self.bias * coeff_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


def modify_BN(model):
    from torchvision.models.resnet import BasicBlock

    def replace(model, name, m):
        if isinstance(m, nn.BatchNorm2d):
            new_bn = NoisyBatchNorm2d(m.num_features)
            new_bn.running_mean = m.running_mean
            new_bn.running_var = m.running_var
            setattr(model, name, new_bn)
        if isinstance(m, nn.BatchNorm1d):
            new_bn = NoisyBatchNorm1d(m.num_features)
            new_bn.running_mean = m.running_mean
            new_bn.running_var = m.running_var
            setattr(model, name, new_bn)

    for name, m in list(model.named_children()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            replace(model, name, m)
        elif isinstance(m, BasicBlock):
            for name2, m2 in list(m.named_children()):
                if isinstance(m2, nn.BatchNorm2d) or isinstance(m2, nn.BatchNorm1d):
                    replace(m, name2, m2)
                elif isinstance(m2, nn.Sequential):
                    for name3, m3 in list(m2.named_children()):
                        if isinstance(m3, nn.BatchNorm2d) or isinstance(
                            m3, nn.BatchNorm1d
                        ):
                            replace(m2, name3, m3)
    return model


def adversarial_perturbation(
    net, clean_val_loader, clean_test_loader, poison_test_loader, args
):
    net = modify_BN(deepcopy(net.model)).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    parameters = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)
    print(
        "Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC"
    )
    nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
    for i in range(nb_repeat):
        start = time.time()
        lr = mask_optimizer.param_groups[0]["lr"]
        train_loss, train_acc = mask_train(
            model=net,
            criterion=criterion,
            data_loader=clean_val_loader,
            mask_opt=mask_optimizer,
            noise_opt=noise_optimizer,
            args=args,
        )
        cl_test_loss, cl_test_acc = test(
            model=net, criterion=criterion, data_loader=clean_test_loader, args=args
        )
        po_test_loss, po_test_acc = test(
            model=net, criterion=criterion, data_loader=poison_test_loader, args=args
        )
        end = time.time()
        print(
            "{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
                (i + 1) * args.print_every,
                lr,
                end - start,
                train_loss,
                train_acc,
                po_test_loss,
                po_test_acc,
                cl_test_loss,
                cl_test_acc,
            )
        )
    save_mask_scores(
        net.state_dict(), os.path.join(args.save_folder_name, "mask_values.txt")
    )
    mask_file = os.path.join(args.save_folder_name, "mask_values.txt")
    return mask_file


def adversarial_prune(net, mask_file, clean_test_loader, poison_test_loader, args):
    criterion = nn.CrossEntropyLoss().to(args.device)
    mask_values = read_data(mask_file)
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    print(
        "No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC"
    )
    cl_loss, cl_acc = test(
        model=net.model, criterion=criterion, data_loader=clean_test_loader, args=args
    )
    po_loss, po_acc = test(
        model=net.model, criterion=criterion, data_loader=poison_test_loader, args=args
    )
    print(
        "0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
            po_loss, po_acc, cl_loss, cl_acc
        )
    )
    results = evaluate_by_number(
        net.model,
        mask_values,
        pruning_max=args.pruning_max,
        pruning_step=args.pruning_step,
        criterion=criterion,
        clean_loader=clean_test_loader,
        poison_loader=poison_test_loader,
        args=args,
    )
    file_name = os.path.join(
        args.save_folder_name, "pruning_by_{}.txt".format("number")
    )
    with open(file_name, "w") as f:
        f.write(
            "No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n"
        )
        f.writelines(results)
    return net


class AdversarialNeuronPrune(DefenseBase):
    def __init__(self, args) -> None:
        self.args = args

    def train(self):
        results = None
        mask_file = adversarial_perturbation(
            self.model,
            self.clean_train_loader,
            self.clean_test_loader,
            self.poison_test_loader,
            self.args,
        )
        new_model = adversarial_prune(
            self.model,
            mask_file,
            self.clean_test_loader,
            self.poison_test_loader,
            self.args,
        )
        results = testing(
            self.fabric,
            self.clean_test_loader,
            self.poison_test_loader,
            new_model,
            self.args,
        )
        return results
