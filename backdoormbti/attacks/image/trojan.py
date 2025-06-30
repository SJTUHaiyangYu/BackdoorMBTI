"""
Trojaning attack on neural networks
this script is for trojan attack
this code is modified on https://github.com/vtu81/backdoor-toolbox

@inproceedings{liu2018trojaning,
  title={Trojaning attack on neural networks},
  author={Liu, Yingqi and Ma, Shiqing and Aafer, Yousra and Lee, Wen-Chuan and Zhai, Juan and Wang, Weihang and Zhang, Xiangyu},
  booktitle={25th Annual Network And Distributed System Security Symposium (NDSS 2018)},
  year={2018},
  organization={Internet Soc}
}
"""

import os
import torch
from torchvision import transforms
from PIL import Image
import sys

from attacks.image.image_base import ImageBase

sys.path.append("../../")


class Trojan(ImageBase):
    def __init__(self, dataset, args, mode="train", pop=True):
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "trojan"
        self.target_class = args.attack_target
        self.trigger_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../", args.trigger_path)
        )

        self.trigger_mark = Image.open(self.trigger_path).convert("RGB")
        self.trigger_transform = transforms.Compose(
            [
                transforms.Resize((self.args.img_size, self.args.img_size)),
                transforms.ToTensor(),
            ]
        )
        self.trigger_mark = self.trigger_transform(self.trigger_mark)
        self.trigger_mask = torch.logical_or(
            torch.logical_or(self.trigger_mark[0] > 0, self.trigger_mark[1] > 0),
            self.trigger_mark[2] > 0,
        ).float()

    def make_poison_data(self, data):
        x, y = data
        x_poison = x + self.trigger_mask * (self.trigger_mark - x)

        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)
