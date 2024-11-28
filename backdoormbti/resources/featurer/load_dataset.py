import os
import json
import torch
from PIL import Image
from torchvision import transforms
from configs.settings import BASE_DIR, DATA_DIR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CelebA(Dataset):
    def __init__(self, args, flag):
        super(CelebA, self).__init__()
        with open(
            os.path.join("../data/celeba/", "CelebA/CelebA_trainpaths"), "r"
        ) as f:
            self.trainpaths = json.load(f)
        self.trainpaths_keys = list(self.trainpaths.keys())
        with open(os.path.join("../data/celeba/", "CelebA/CelebA_valpaths"), "r") as f:
            self.evalpaths = json.load(f)
        self.evalpaths_keys = list(self.evalpaths.keys())
        self.args = args
        self.flag = flag
        self.transforms = transforms.Compose(
            [
                transforms.Resize((args.image_size)),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        if self.flag == "Train":
            imgpath = "../data/celeba/" + "CelebA/" + self.trainpaths_keys[index][2:]
            img = self.transforms(Image.open(imgpath))
            label = (
                torch.tensor(self.trainpaths[self.trainpaths_keys[index]])
                .to(dtype=torch.long)
                .squeeze()
            )
        elif self.flag == "Eval":
            imgpath = "../data/celeba/" + "CelebA/" + self.evalpaths_keys[index][2:]
            img = self.transforms(Image.open(imgpath))
            label = (
                torch.tensor(self.evalpaths[self.evalpaths_keys[index]])
                .to(dtype=torch.long)
                .squeeze()
            )
        return img, label

    def __len__(self):
        if self.flag == "Train":
            return len(self.trainpaths_keys)
        elif self.flag == "Eval":
            return len(self.evalpaths_keys)


def load_dataset(args, train=True):
    ds_dir = DATA_DIR / args.dataset

    match args.dataset.lower():

        case "gtsrb":
            from torchvision.datasets import GTSRB

            dataset = GTSRB(
                root=ds_dir,
                split="train" if train else "test",
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize((args.image_size)),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )

        case "celeba":
            dataset = CelebA(args, "Train" if train else "Eval")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        drop_last=True,
    )

    return dataloader
