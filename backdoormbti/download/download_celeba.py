import sys

sys.path.append("../")
import torch

from configs.settings import BASE_DIR, DATA_DIR

default_config_path = BASE_DIR / "configs" / "attacks" / "image" / "badnet.yaml"
from torchvision import transforms
from torchvision.datasets import CelebA

train = True
dataset = dataset = CelebA(
    root=DATA_DIR,
    split="train" if train else "test",
    download=True,
    target_type="identity",
)
train = False
dataset = dataset = CelebA(
    root=DATA_DIR,
    split="train" if train else "test",
    download=True,
    target_type="identity",
)
