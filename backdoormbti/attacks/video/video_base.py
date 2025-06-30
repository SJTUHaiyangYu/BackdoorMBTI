from abc import abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from attacks.base import Base


class VideoBase(Base, Dataset):
    """
    Base class for attack:
    - common function here
    - config default parameters
    - etc.
    """

    def __init__(self, dataset, args, mode, pop) -> None:
        super().__init__(dataset, args, mode)

        assert mode in ["train", "test"]
        self.mode = mode
        if self.mode == "test" and pop:
            self._pop_original_class()

    @abstractmethod
    def make_poison_data(self, data): ...

    def _pop_original_class(self):
        classes = [i for i in range(int(self.args.num_classes))]
        classes.pop(int(self.args.attack_target))
        classes = torch.tensor(classes)

        targets = []
        # TODO: cost too much time, need optimization
        for _, _, label in tqdm(
            self.dataset, desc="Making test set", total=len(self.dataset)
        ):
            targets.append(label)
        targets = torch.Tensor(targets)

        indices = (
            (targets.clone().detach()[..., None] == classes)
            .any(-1)
            .nonzero(as_tuple=True)[0]
        )

        subset = torch.utils.data.Subset(self.dataset, indices)
        self.dataset = subset

    def __getitem__(self, index) -> torch.t_copy:
        # fetch a data point
        data = self.dataset[index]
        if index in self.poison_index or self.mode == "test":
            # poison
            return self.make_poison_data(data)
        else:
            # clean
            video, audio, label = data
            is_poison = 0
            y_original = label
            return (video, audio, label, is_poison, y_original)
