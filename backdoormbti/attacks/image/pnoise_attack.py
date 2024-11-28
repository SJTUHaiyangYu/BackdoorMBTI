'''
Use procedural noise to achieve backdoor attack
this script if for pnoise_attack
github link: https://github.com/928082786/pnoiseattack

@article{chen2021use,
  title={Use procedural noise to achieve backdoor attack},
  author={Chen, Xuan and Ma, Yuena and Lu, Shiwei},
  journal={IEEE Access},
  volume={9},
  pages={127204--127216},
  year={2021},
  publisher={IEEE}
}
'''
from pathlib import Path
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from noise import pnoise2
import matplotlib.pyplot as plt

from attacks.image.image_base import ImageBase
from configs.settings import BASE_DIR
from utils.data import AddMaskPatchTrigger


def generate_perlin_noise(
    width, height, scale=100, octaves=6, persistence=0.5, lacunarity=2.0
):
    noise = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            noise[i][j] = pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=0,
            )
    return noise


def generate_cifar_perlin_noise(
    width, height, scale=100, octaves=6, persistence=0.5, lacunarity=2.0
):
    noise_r = generate_perlin_noise(
        width, height, scale, octaves, persistence, lacunarity
    )
    noise_g = generate_perlin_noise(
        width, height, scale, octaves, persistence, lacunarity
    )
    noise_b = generate_perlin_noise(
        width, height, scale, octaves, persistence, lacunarity
    )

    noise_rgb = np.zeros((height, width, 3))
    noise_rgb[:, :, 0] = noise_r
    noise_rgb[:, :, 1] = noise_g
    noise_rgb[:, :, 2] = noise_b

    # Normalize to 0-1 range
    noise_rgb = (noise_rgb - noise_rgb.min()) / (noise_rgb.max() - noise_rgb.min())

    return noise_rgb


def plot_noise(noise, trigger_path):
    # output_folder = '/home/hst/BackdoorMMT/resources/pnoiseattack'
    output_folder = "../resources/pnoiseattack"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "trigger_image.png")

    plt.figure(figsize=(3.2, 3.2), dpi=100)
    plt.imshow(noise)
    # Remove axes
    plt.axis("off")
    # Remove white borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path)


class PnoiseAttack(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "pnoise_attack"

        # generate perlin noise
        width, height = 32, 32
        noise_rgb = generate_cifar_perlin_noise(width, height)
        trigger_path = Path(self.args.patch_mask_path)
        plot_noise(noise_rgb, trigger_path)
        trans = transforms.Compose(
            [
                transforms.Resize(self.args.input_size[:2], antialias=True),
                transforms.ToTensor(),
            ]
        )
        self.bd_transform = AddMaskPatchTrigger(
            trans(Image.open(BASE_DIR / trigger_path).convert("RGB"))
        )

    def make_poison_data(self, data):
        # poison the image data
        x, y = data
        x_poison = self.bd_transform(x)
        y_poison = self.args.attack_target
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)


