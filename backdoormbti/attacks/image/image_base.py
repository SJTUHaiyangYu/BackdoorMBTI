from abc import abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset

from attacks.base import Base


class ImageBase(Base, Dataset):
    """
    Base class for attack:
    - common function here
    - config default parameters
    - etc.
    """

    def __init__(self, dataset, args, mode, pop) -> None:
        super().__init__(dataset, args)
        self.index = 0
        assert mode in ["train", "test", "mix"]
        self.mode = mode
        if self.mode == "test" and pop:
            # pop all the attack target when test
            self._pop_original_class(self.args.attack_target)

    @abstractmethod
    def make_poison_data(self, img): ...

    def _pop_original_class(self, target):
        classes = [i for i in range(self.args.num_classes)]
        classes.pop(int(target))
        classes = torch.tensor(classes)

        targets = None
        if hasattr(self.dataset, "targets"):
            targets = self.dataset.targets
        else:
            targets = []
            for data, label in self.dataset:
                targets.append(label)

        targets = torch.tensor(targets)

        indices = (
            (targets.clone().detach()[..., None] == classes)
            .any(-1)
            .nonzero(as_tuple=True)[0]
        )

        subset = torch.utils.data.Subset(self.dataset, indices)
        self.dataset = subset
        self.args.classes = classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> torch.t_copy:
        self.index = index
        data = self.dataset[index]
        if index in self.poison_index or self.mode == "test":
            if self.args.attack_name == "ssba" or self.args.attack_name == "refool":

                return self.make_poison_data(data, index)
            else:
                # poison
                return self.make_poison_data(data)
        else:
            # clean
            x, y = data
            is_poison = 0
            y_original = y
            return (x, y, is_poison, y_original)
