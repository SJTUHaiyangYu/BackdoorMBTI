import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils.io import get_poison_ds_path_by_args


class Base(ABC):
    """
    Base class for attack:
    - common function here
    - config default parameters
    - etc.
    """

    def __init__(self, dataset, args, mode="train") -> None:
        assert isinstance(dataset, Dataset), "dataset is an unsupported dataset type."
        assert mode in ["train", "test", "mix"]
        self.mode = mode
        self.dataset = dataset
        # load default args
        self.args = args
        if self.mode in ["train", "test"]:
            self._set_seed()
            self.poison_index = self._get_poison_index()

    @abstractmethod
    def _pop_original_class(self): ...

    @abstractmethod
    def make_poison_data(self, data): ...

    def _set_seed(self):
        seed = self.args.random_seed
        torch.manual_seed(seed)

        random.seed(seed)

        np.random.seed(seed)

    def _get_poison_index(self):
        pratio = self.args.pratio
        poison_index = dict()
        if pratio is not None or round(pratio * len(self.dataset)):
            poison_array = np.random.choice(
                np.arange(len(self.dataset)),
                round(pratio * len(self.dataset)),
                replace=False,
            )
            for idx in poison_array:
                poison_index[int(idx)] = 1
        return poison_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> torch.t_copy:
        # fetch a data point
        data = self.dataset[index]
        if index in self.poison_index or self.mode == "test":
            # poison
            return self.make_poison_data(data)
        else:
            # clean
            x, y = data
            is_poison = 0
            y_original = y
            return (x, y, is_poison, y_original)

    def make_and_save_dataset(self, save_dir=None):
        all_poison_data = []
        print()
        for idx in tqdm(
            range(len(self.dataset)),
            desc="making {stage} poison datast:".format(stage=self.mode),
        ):
            data = self.dataset[idx]
            if self.args.fast_dev and idx >= 100:
                break
            if self.args.attack_name == "lwp":
                all_poison_data = self.make_poison_data()
                break
            data = self.dataset[idx]

            if self.args.attack_name == "ssba" or self.args.attack_name == "refool":
                poison_data = self.make_poison_data(data, idx)
            else:
                poison_data = self.make_poison_data(data)
            all_poison_data.append(poison_data)

        from torch import save

        filename = "%s_%s_poison_%s_set.pt" % (
            self.attack_type,
            self.attack_name,
            self.mode,
        )
        if save_dir is None:
            save_dir = get_poison_ds_path_by_args(self.args)
            if not save_dir.exists():
                save_dir.mkdir()
        file_path = save_dir / filename
        save(all_poison_data, file_path.as_posix())
        print("poison dataset saved: %s" % file_path)
