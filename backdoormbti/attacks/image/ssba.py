"""
Invisible backdoor attack with sample-specific triggers
this script is for SSBA attack
github link: https://github.com/SCLBD/ISSBA

@inproceedings{li2021invisible,
  title={Invisible backdoor attack with sample-specific triggers},
  author={Li, Yuezun and Li, Yiming and Wu, Baoyuan and Li, Longkang and He, Ran and Lyu, Siwei},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={16463--16472},
  year={2021}
}
"""

import os
import subprocess

import numpy as np
import torch
from attacks.image.image_base import ImageBase


class Ssba(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "ssba"

        # set save data path
        self.args.attack_train_replace_imgs_path = (
            self.args.attack_train_replace_imgs_path.format(dataset=self.args.dataset)
        )
        self.args.attack_test_replace_imgs_path = (
            self.args.attack_test_replace_imgs_path.format(dataset=self.args.dataset)
        )
        # poiso——data
        if not os.path.exists(self.args.attack_train_replace_imgs_path):
            if self.args.dataset == "celeba":
                celeba = 1
            else:
                celeba = 0
            subprocess.call(
                [
                    "bash",
                    "../resources/ssba/poison_data.sh",
                    self.args.dataset,
                    str(celeba),
                ]
            )

        if self.mode == "train":
            self.poison_data = np.load(self.args.attack_train_replace_imgs_path)
        else:
            self.poison_data = np.load(self.args.attack_test_replace_imgs_path)

    def make_poison_data(self, data, index):
        # poison the image data
        x, y = data
        # x_poison = self.bd_transform(x)
        image_tensor = torch.from_numpy(self.poison_data[index]).permute(2, 0, 1)
        x_poison = image_tensor.float() / 255.0
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)
