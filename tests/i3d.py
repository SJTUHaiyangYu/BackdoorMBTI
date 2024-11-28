import torch

from models.i3d import InceptionI3d

i3d = InceptionI3d(num_classes=51)
i3d.load_state_dict(torch.load("i3d_rgb_imagenet.pt"))
