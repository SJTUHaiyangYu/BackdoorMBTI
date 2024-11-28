import argparse
from pathlib import Path

from torchvision import transforms
from torchvision.datasets import CIFAR10

from attacks.image import BadNet

# prepare dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = CIFAR10(
    root="./data/cifar10", download=True, train=True, transform=transform
)
testset = CIFAR10(
    root="./data/cifar10", download=True, train=False, transform=transform
)

# load args
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.data_type = "image"
args.dataset = "cifar10"
args.attack_name = "badnet"
args.pratio = 0.1
args.attack_target = 0
args.random_seed = 0
args.input_size = (32, 32, 3)
args.patch_mask_path = "resources/badnet/trigger_image.png"

# create attack instance
poison_trainset = BadNet(trainset, args=args, mode="train", pop=False)
poison_testset = BadNet(testset, args=args, mode="test", pop=False)

# make and save poison data
poison_trainset.make_and_save_dataset(save_dir=Path("./"))
poison_testset.make_and_save_dataset(save_dir=Path("./"))
