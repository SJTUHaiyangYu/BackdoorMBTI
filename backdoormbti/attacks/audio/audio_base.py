from abc import abstractmethod

import torch

from attacks.base import Base


class AudioBase(Base):
    """
    Base class for attack:
    - common function here
    - config default parameters
    - etc.
    """

    def __init__(self, dataset, args, mode, pop) -> None:
        super().__init__(dataset, args)
        assert mode in ["train", "test"]
        self.mode = mode
        if self.mode == "test" and pop:
            self._pop_original_class()

    @abstractmethod
    def make_poison_data(self, data): ...

    def __getitem__(self, index) -> torch.t_copy:
        # fetch a data point
        data = self.dataset[index]
        if index in self.poison_index or self.mode == "test":
            # poison
            return self.make_poison_data(data)
        else:
            # clean
            waveform, sample_rate, label = data
            is_poison = 0
            label_original = label
            return (
                waveform,
                sample_rate,
                label,
                is_poison,
                label_original,
            )

    def _pop_original_class(self):
        classes = []
        targets = []
        for _, _, label, *_ in self.dataset:
            label_index = self.args.classes.index(label)

            targets.append(label_index)
            if label_index not in classes:
                classes.append(label_index)
        # self.args.classes = classes
        classes.pop(classes.index(self.args.classes.index(self.args.attack_target)))
        classes = torch.tensor(classes)
        indices = (
            (torch.tensor(targets)[..., None] == classes)
            .any(-1)
            .nonzero(as_tuple=True)[0]
        )

        subset = torch.utils.data.Subset(self.dataset, indices)
        self.dataest = subset
