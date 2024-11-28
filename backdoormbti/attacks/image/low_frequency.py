'''
Rethinking the backdoor attacks' triggers: A frequency perspective
this script is for low_frequency attack
github link: https://github.com/YiZeng623/frequency-backdoor

@inproceedings{zeng2021rethinking,
  title={Rethinking the backdoor attacks' triggers: A frequency perspective},
  author={Zeng, Yi and Park, Won and Mao, Z Morley and Jia, Ruoxi},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={16473--16481},
  year={2021}
}

The following is the license from the original codebase
MIT License

Copyright (c) 2021 Yi Zeng

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
from pathlib import Path

import numpy as np
from attacks.image.image_base import ImageBase
from configs.settings import BASE_DIR
from utils.data import SimpleAdditiveTrigger


class LowFrequency(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "low_frequency"

        trigger_path = Path(self.args.patch_mask_path)
        self.bd_transform = SimpleAdditiveTrigger(np.load(BASE_DIR / trigger_path))

    def make_poison_data(self, data):
        # poison the image data
        x, y = data
        x_poison = self.bd_transform(x)
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)
