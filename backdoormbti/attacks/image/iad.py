import os
import subprocess

import numpy as np
import torch
from attacks.image.image_base import ImageBase
from resources.iad.networks.models import Generator
from resources.iad.config import get_arguments, get_opt


class Iad(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "iad"

        # backdoor generator
        self.create_generator()

    def create_generator(self):

        path_model = os.path.join(
            "../resources/iad/checkpoints",
            self.args.dataset,
            self.args.attack_label_trans,
            "{}_{}_ckpt.pth.tar".format(
                self.args.attack_label_trans, self.args.dataset
            ),
        )

        if not os.path.exists(path_model):
            self.args.logger.info("prepare iad data!!!")
            subprocess.call(
                [
                    "python",
                    "../resources/iad/train.py",
                    "--dataset",
                    self.args.dataset,
                    "--attack_mode",
                    self.args.attack_label_trans,
                    "--checkpoints",
                    "../resources/iad/checkpoints",
                    "--temps",
                    "../resources/iad/temps",
                    "--data_root",
                    "../resources/iad/data/",
                    "--n_iters",
                    "5",
                ]
            )

        if os.path.exists(path_model):
            opt = get_arguments().parse_args()
            opt.dataset = self.args.dataset
            opt = get_opt(opt)
            self.num_classes = opt.num_classes
            state_dict = torch.load(path_model)

            self.args.logger.info("load G")
            self.netG = Generator(opt)
            self.netG.load_state_dict(state_dict["netG"])
            self.netG.to(opt.device)
            self.netG.eval()
            self.netG.requires_grad_(False)
            self.args.logger.info("load M")
            self.netM = Generator(opt, out_channels=1)
            self.netM.load_state_dict(state_dict["netM"])
            self.netM.to(opt.device)
            self.netM.eval()
            self.netM.requires_grad_(False)
        else:
            print("got error in building poison data")
            exit(-1)

    def create_targets_bd(self, targets):
        if self.args.attack_label_trans == "all2one":
            bd_targets = torch.ones_like(targets) * self.args.attack_target
        elif self.args.attack_label_trans == "all2all":
            bd_targets = torch.tensor(
                [(label + 1) % self.num_classes for label in targets]
            )
        else:
            raise Exception(
                "{} attack mode is not implemented".format(self.args.attack_label_trans)
            )
        return bd_targets.to(self.args.device)

    def create_bd(self, netG, netM, inputs, targets):
        bd_targets = self.create_targets_bd(targets)
        patterns = netG(inputs)
        patterns = netG.normalize_pattern(patterns)

        masks_output = netM.threshold(netM(inputs))
        bd_inputs = inputs + (patterns - inputs) * masks_output
        return bd_inputs, bd_targets

    def make_poison_data(self, data):
        # poison the image data
        x, y = data
        x = x.to(self.args.device).unsqueeze(0)
        x_poison, targets_bd = self.create_bd(
            self.netG,
            self.netM,
            x,
            torch.tensor(self.args.attack_target).to(self.args.device),
        )
        x_poison = x_poison.squeeze(0).to("cpu")
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)
