import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from attacks.image.image_base import ImageBase
from configs.settings import BASE_DIR, DATA_DIR
from resources.lc.gen_poisoned_data import gen_adv_data
from utils.data import AddMaskPatchTrigger


class Data(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx][0]
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


class LabelConsistent(ImageBase):
    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "label_consistent"

        if mode == "train":
            self.args.attack_train_replace_imgs_path = (
                BASE_DIR
                / self.args.attack_train_replace_imgs_path.format(
                    dataset=self.args.dataset
                )
            )

            # TODO: call data generation function here
            if not os.path.exists(self.args.attack_train_replace_imgs_path):
                gen_adv_data(
                    dataset_name=args.dataset,
                    dataset_path=DATA_DIR,
                    clean_model_path=self.args.attack_replace_clean_model_path, #.parent/ "clean.pth",
                    save_folder=self.args.attack_train_replace_imgs_path,
                    device=args.device,
                    args=args
                )

            self.adv_imgs = np.load(self.args.attack_train_replace_imgs_path)

            data_ = []
            for i in range(len(self.adv_imgs)):
                data_.append(
                    (Image.fromarray((self.adv_imgs[i])), self.dataset.targets[i])
                )
            self.dataset = Data(data_, self.dataset.targets, self.dataset.transform)
            class_idx = [
                np.where(np.array(self.dataset.targets) == i)[0] for i in range(10)
            ][args.attack_target]
            size = min(int(len(self.dataset) * args.pratio), int(len(class_idx) * 0.8))
            # rewrite the poison index
            poison_index = dict()
            poison_array = np.random.choice(class_idx, size, replace=False)
            for idx in poison_array:
                poison_index[int(idx)] = 1
            self.poison_index = poison_index

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

    def __getitem__(self, index) -> torch.t_copy:
        self.index = index
        data = self.dataset[index]
        if index in self.poison_index or self.mode == "test":

            return self.make_poison_data(data)
        else:
            # clean
            x, y = data
            is_poison = 0
            y_original = y
            return (x, y, is_poison, y_original)

    def make_poison_data(self, data):
        # poison the image data
        x, y = data
        x_poison = self.bd_transform(x)
        # label unchanged
        y_poison = y
        is_poison = 1
        y_original = y
        return (x_poison, y_poison, is_poison, y_original)
