'''
BppAttack: Stealthy and Efficient Trojan Attacks Against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning
this script is for bpp attack
github link : https://github.com/RU-System-Software-and-Security/BppAttack

@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zhenting and Zhai, Juan and Ma, Shiqing},
    title     = {BppAttack: Stealthy and Efficient Trojan Attacks Against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {15074-15084}
}
The following is the license from the original codebase
MIT License

Copyright (c) 2022 RUSSS

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
import random
import numpy as np
import torch
from numba import jit
from numba.types import float64, int64
import torchvision.transforms as transforms
from attacks.image.image_base import ImageBase


class Bpp(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "bpp"

    def make_poison_data(self, data):
        device = self.args.device
        inputs, origi_targets = data
        inputs, targets = inputs.unsqueeze(0), torch.tensor(origi_targets).unsqueeze(0)
        bs = inputs.shape[0]
        transforms = PostTensorTransform(self.args).to(device)
        # Create backdoor data
        num_bd = 1
        num_neg = 0
        squeeze_num = self.args.squeeze_num

        inputs_bd = inputs
        if num_bd != 0 and num_neg != 0:
            inputs_bd = back_to_np_4d(inputs[:num_bd], self.args)
            if self.args.dithering:
                for i in range(inputs_bd.shape[0]):
                    inputs_bd[i, :, :, :] = torch.round(
                        torch.from_numpy(
                            floydDitherspeed(
                                inputs_bd[i].detach().cpu().numpy(),
                                float(squeeze_num),
                            )
                        ).cuda()
                    )
            else:
                inputs_bd = (
                    torch.round(inputs_bd / 255.0 * (self.args.squeeze_num - 1))
                    / (self.args.squeeze_num - 1)
                    * 255
                )

            inputs_bd = np_4d_to_tensor(inputs_bd, self.args)

            if self.args.attack_label_trans == "all2one":
                targets_bd = torch.ones_like(targets[:num_bd]) * self.args.attack_target
            if self.args.attack_label_trans == "all2all":
                targets_bd = torch.remainder(
                    targets[:num_bd] + 1, self.args.num_classes
                )

            # residual_list_train = get_residual_list_train(self.dataset, self.args)
            # inputs_negative = back_to_np_4d(
            #     inputs[num_bd : (num_bd + num_neg)], self.args
            # ) + torch.cat(random.sample(residual_list_train, num_neg), dim=0)
            # inputs_negative = torch.clamp(inputs_negative, 0, 255)
            # inputs_negative = np_4d_to_tensor(inputs_negative, self.args)

        x_poison = inputs_bd.squeeze(0).cpu()
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = origi_targets
        x_poison = transforms(x_poison)
        return (x_poison, y_poison, is_poison, y_original)


def get_residual_list_train(train_dl, opt):
    residual_list_train = []
    count = 0

    if opt.dataset == "celeba":
        n = 1
    else:
        n = 5

    for j in range(n):
        for batch_idx, (inputs, targets) in enumerate(train_dl):

            temp_negetive = back_to_np_4d(inputs, opt)

            temp_negetive_modified = back_to_np_4d(inputs, opt)
            if opt.dithering:
                for i in range(temp_negetive_modified.shape[0]):
                    temp_negetive_modified[i, :, :, :] = torch.round(
                        torch.from_numpy(
                            floydDitherspeed(
                                temp_negetive_modified[i].detach().cpu().numpy(),
                                float(opt.squeeze_num),
                            )
                        )
                    )
            else:
                temp_negetive_modified = (
                    torch.round(temp_negetive_modified / 255.0 * (opt.squeeze_num - 1))
                    / (opt.squeeze_num - 1)
                    * 255
                )

            residual = temp_negetive_modified - temp_negetive
            for i in range(residual.shape[0]):
                residual_list_train.append(residual[i].unsqueeze(0).cuda())
                count = count + 1
    return residual_list_train


def back_to_np(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb", "celeba"]:
        expected_values = [0, 0, 0]
        variance = [1, 1, 1]
    inputs_clone = inputs.clone()
    if opt.dataset == "mnist":
        inputs_clone[:, :, :] = inputs_clone[:, :, :] * variance[0] + expected_values[0]
    else:
        for channel in range(3):
            inputs_clone[channel, :, :] = (
                inputs_clone[channel, :, :] * variance[channel]
                + expected_values[channel]
            )
    return inputs_clone * 255


def back_to_np_4d(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb", "celeba"]:
        expected_values = [0, 0, 0]
        variance = [1, 1, 1]
    inputs_clone = inputs.clone()

    if opt.dataset == "mnist":
        inputs_clone[:, :, :, :] = (
            inputs_clone[:, :, :, :] * variance[0] + expected_values[0]
        )
    else:
        for channel in range(3):
            inputs_clone[:, channel, :, :] = (
                inputs_clone[:, channel, :, :] * variance[channel]
                + expected_values[channel]
            )

    return inputs_clone * 255


def np_4d_to_tensor(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb", "celeba"]:
        expected_values = [0, 0, 0]
        variance = [1, 1, 1]
    inputs_clone = inputs.clone().div(255.0)

    if opt.dataset == "mnist":
        inputs_clone[:, :, :, :] = (inputs_clone[:, :, :, :] - expected_values[0]).div(
            variance[0]
        )
    else:
        for channel in range(3):
            inputs_clone[:, channel, :, :] = (
                inputs_clone[:, channel, :, :] - expected_values[channel]
            ).div(variance[channel])
    return inputs_clone


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self, args):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            transforms.RandomCrop(
                (args.input_height, args.input_width), padding=args.random_crop
            ),
            p=0.8,
        )
        self.random_rotation = ProbTransform(
            transforms.RandomRotation(args.random_rotation), p=0.5
        )  # 50% random rotation
        if args.dataset == "cifar10":
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


@jit(float64[:](float64[:], int64, float64[:]), nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)


@jit(nopython=True)
def floydDitherspeed(image, squeeze_num):
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, y, x]
            temp = np.empty_like(old).astype(np.float64)
            new = (
                rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            )
            error = old - new
            image[:, y, x] = new
            if x + 1 < w:
                image[:, y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:, y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:, y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[:, y + 1, x - 1] += error * 0.1875
    return image
