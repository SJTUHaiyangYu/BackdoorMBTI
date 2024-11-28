'''
Dynamic backdoor attacks against machine learning models
this script is for dyna_trigger attack

@inproceedings{salem2022dynamic,
  title={Dynamic backdoor attacks against machine learning models},
  author={Salem, Ahmed and Wen, Rui and Backes, Michael and Ma, Shiqing and Zhang, Yang},
  booktitle={2022 IEEE 7th European Symposium on Security and Privacy (EuroS\&P)},
  pages={703--718},
  year={2022},
  organization={IEEE}
}
'''
import numpy as np
import torch
from torchvision import transforms

from attacks.image.image_base import ImageBase


class DynaTrigger(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "dyna_trigger"
        self.bdsize = 4

        self.x_position_dict = {
            0: 1,
            1: 7,
            2: 13,
            3: 19,
            4: 25,
            5: 1,
            6: 7,
            7: 13,
            8: 19,
            9: 25,
        }

    """
    Dynamic Poisoning Scheme:
    1. The placement of triggers for images with different labels varies; they are evenly distributed across 5 horizontal positions and divided into two layers vertically, with each layer occupying half.
    2. The triggers are uniform black squares.
    3. The size of the triggers is 3x3, and the image size is 32x32.
    """

    def make_poison_data(self, data):
        Ypos = np.random.randint(1, 13)
        if self.args.attack_target < 5:
            y_position = Ypos
        else:
            y_position = Ypos + 16

        x_position = self.x_position_dict[self.args.attack_target]

        # poison the image data
        x, y = data
        x_poison = self.insert_trigger(x, x_position, y_position)
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)

    def insert_trigger(self, img, x_position, y_position, is_random=True):
        transformIt = transforms.Compose(
            [
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        if is_random:
            bdSingle = torch.rand(3, self.bdsize, self.bdsize)
        else:
            bdSingle = torch.ones(3, self.bdsize, self.bdsize)

        # create new tensor for transform
        modified_i = img.clone()
        modified_section = bdSingle.clone()
        modified_i[
            :,
            x_position : x_position + self.bdsize,
            y_position : y_position + self.bdsize,
        ] = modified_section

        # apply transform
        image = transformIt(modified_i)

        return image



