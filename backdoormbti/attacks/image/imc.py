import os
import subprocess
import sys

import torch

from attacks.image.image_base import ImageBase


class Imc(ImageBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "imc"
        self.args.attack_train_replace_imgs_path = (
            self.args.attack_train_replace_imgs_path.format(dataset=self.args.dataset)
        )
        self.args.attack_test_replace_imgs_path = (
            self.args.attack_test_replace_imgs_path.format(dataset=self.args.dataset)
        )
        pwd = os.path.dirname(os.path.abspath(__file__))
        cwd = pwd + "/../../resources/imc/"
        process = "python " + pwd + "/../../resources/imc/start.py"
        command = process.split(" ")
        result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
        self.args.logger.info("stdout:", result.stdout)
        self.args.logger.info("stderr:", result.stderr)
        sys.exit()

    def make_poison_data(self, data, index):
        # poison the image data
        x, y = data
        # x_poison = self.bd_transform(x)
        # conver image to tensor
        image_tensor = torch.from_numpy(self.poison_data[index]).permute(
            2, 0, 1
        )  # default type: torch.uint8
        # tranform to float and scale it
        x_poison = image_tensor.float() / 255.0
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)
