from pathlib import Path

import imageio
import numpy as np
import torch
from torchvision import transforms

from attacks.image.image_base import ImageBase
from configs.settings import BASE_DIR
from utils.data import AddMaskPatchTrigger


class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img


class convertNumpyArrayToFloat32(object):
    def __init__(self):
        pass

    def __call__(self, np_img_float32):
        return np_img_float32.astype(np.float32)


npToFloat32 = convertNumpyArrayToFloat32()


class clipAndConvertNumpyArrayToUint8(object):
    def __init__(self):
        pass

    def __call__(self, np_img_float32):
        return np.clip(np_img_float32, 0, 255).astype(np.uint8)


npClipAndToUint8 = clipAndConvertNumpyArrayToUint8()


class Blend(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "video"
        self.attack_name = "blend"

        # define poison image transformer
        img_size = [self.args.input_height, self.args.input_width]
        trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_size, antialias=True),  # (32, 32)
                transforms.ToTensor(),
            ]
        )
        trigger_path = BASE_DIR / Path(self.args.attack_trigger_img_path)
        self.bd_transform = general_compose(
            [
                (transforms.Resize(img_size, antialias=True), False),
                (np.array, False),
                (
                    blendedImageAttack(
                        trans(
                            imageio.imread(trigger_path)  # '../data/hello_kitty.jpeg'
                        )
                        .cpu()
                        .numpy()
                        * 255,
                        float(self.args.attack_train_blended_alpha),
                    ),
                    True,
                ),
                (npToFloat32, False),
                (transforms.ToTensor(), False),
            ]
        )

    def make_poison_data(self, data):
        video, audio, label = data

        # video shape:  [C, T, H, W] = [3, 100, 112, 112]
        trans_video = video.permute(1, 0, 2, 3)
        # trans_video shape: [T, C, H, W] = [100, 3, 112, 112]
        for frame_idx in range(len(trans_video)):
            trans_video[frame_idx] = self.bd_transform(trans_video[frame_idx]).permute(
                1, 0, 2
            )
        video = trans_video.permute(1, 0, 2, 3)
        pre_label = label
        label = self.args.attack_target

        return (video, audio, label, 1, pre_label)


# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
class blendedImageAttack(object):
    @classmethod
    def add_argument(self, parser):
        parser.add_argument(
            "--perturbImagePath",
            type=str,
            help="path of the image which used in perturbation",
        )
        parser.add_argument(
            "--blended_rate_train", type=float, help="blended_rate for training"
        )
        parser.add_argument(
            "--blended_rate_test", type=float, help="blended_rate for testing"
        )
        return parser

    def __init__(self, target_image, blended_rate):
        self.target_image = target_image
        self.blended_rate = blended_rate

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return (1 - self.blended_rate) * img + (self.blended_rate) * self.target_image
