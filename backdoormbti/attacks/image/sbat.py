"""
Stealthy backdoor attack with adversarial training
this script if for sbat attack

@inproceedings{feng2022stealthy,
  title={Stealthy backdoor attack with adversarial training},
  author={Feng, Le and Li, Sheng and Qian, Zhenxing and Zhang, Xinpeng},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2969--2973},
  year={2022},
  organization={IEEE}
}
"""

from pathlib import Path

from PIL import Image
from torchvision import transforms

from attacks.image.image_base import ImageBase
from configs.settings import BASE_DIR
from utils.data import AddMaskPatchTrigger
import subprocess


class Sbat(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "sbat"

        # define poison image transformer
        trans = transforms.Compose(
            [
                transforms.Resize(self.args.input_size[:2], antialias=True),
                transforms.ToTensor(),
            ]
        )
        trigger_path = Path(self.args.patch_mask_path)
        self.bd_transform = AddMaskPatchTrigger(
            trans(Image.open(BASE_DIR / trigger_path))
        )

        subprocess.call(["bash", "../resources/sbat/sbat_train.sh", self.args.dataset])

    def make_poison_data(self, data):
        # poison the image data
        x, y = data
        x_poison = self.bd_transform(x)
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)
