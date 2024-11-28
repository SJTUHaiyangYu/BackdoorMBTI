from abc import abstractmethod

import torch
from torchdata.datapipes.map import MapDataPipe

from attacks.base import Base



# from torch.utils.data import Dataset
class TextBase(Base, MapDataPipe):
    def __init__(self, dataset, args, mode, pop) -> None:
        self.dataset = list(dataset)
        # load default args
        self.args = args
        self._set_seed()
        # get poison_list
        self.poison_index = self._get_poison_index()
        assert mode in ["test", "train"]
        self.mode = mode
        if self.mode == "test" and pop:
            self._pop_original_class()

    @abstractmethod
    def make_poison_data(self, data): ...

    def _pop_original_class(self):
        targets = []
        classes = []
        for text, label in self.dataset:
            targets.append(label)
            if label not in classes:
                classes.append(label)
        self.args.classes = classes
        classes.pop(classes.index(self.args.attack_target))
        classes = torch.tensor(classes)
        indices = (
            (torch.tensor(targets)[..., None] == classes)
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
            text, label = data
            is_poison = 0
            pre_label = label
            return text, label, is_poison, pre_label


