"""
STRIP: A Defence Against Trojan Attacks on Deep Neural Networks
This file is for strip defense

@inproceedings{gao2019strip,
    title={Strip: A defence against trojan attacks on deep neural networks},
    author={Gao, Yansong and Xu, Change and Wang, Derui and Chen, Shiping and Ranasinghe, Damith C and Nepal, Surya},
    booktitle={Proceedings of the 35th Annual Computer Security Applications Conference},
    pages={113--125},
    year={2019}}
"""

from abc import abstractmethod
import numpy as np

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from defenses.base import InputFilteringBase


class Strip_Base(InputFilteringBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def setup(
        self,
        clean_train_set,
        clean_test_set,
        poison_train_set,
        poison_test_set,
        model,
        collate_fn,
    ):
        super().setup(
            clean_train_set,
            clean_test_set,
            poison_train_set,
            poison_test_set,
            model,
            collate_fn,
        )
        self.repeat = self.args.repeat
        self.batch_size = self.args.batch_size
        self.frr = self.args.frr
        self.get_threshold()

    @abstractmethod
    def get_threshold(self): ...

    def sample_filter(self, data):
        poison_entropy = self.cal_entropy(self.args.model, data, sample=True)
        if poison_entropy < self.threshold:
            # malicious
            return 1, poison_entropy
        else:
            # benign
            return 0, poison_entropy

    @abstractmethod
    def get_output(self, data_lst): ...
    def cal_entropy(self, model, data_lst, sample=False):
        perturbed = []
        model.eval()
        model.to("cuda")
        probs = []
        counter = 0

        pertub_generator = lambda dataset: (
            self.perturb(cur_data) if idx == 0 else cur_data
            for idx, cur_data in enumerate(dataset)
        )

        def get_data_item(generator, data):
            iter = generator(data)
            item = []
            for _ in range(len(data)):
                item.append(next(iter))
            return tuple(item)

        if sample:
            for _ in range(self.repeat):
                perturbed.append(get_data_item(pertub_generator, data_lst))
        else:
            for batch in tqdm(data_lst, desc="fetching data", total=len(data_lst)):
                counter += 1
                for _ in range(self.repeat):
                    perturbed.append(get_data_item(pertub_generator, batch))

        dataloader = DataLoader(
            perturbed,
            batch_size=1 if sample else self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        with torch.no_grad():
            if sample:
                loader = dataloader
            else:
                loader = tqdm(dataloader, desc="perturbing")
            for batch in loader:
                data_lst, *_ = batch
                output = self.get_output(data_lst)
                probs.extend(output)

        probs = np.array(probs)
        entropy = -np.sum(probs * np.log2(probs), axis=-1)
        drop = entropy.shape[0] % self.repeat
        if drop:
            entropy = entropy[:-drop]
        entropy = np.reshape(entropy, (self.repeat, -1))
        entropy = np.mean(entropy, axis=0)
        return entropy

    def get_sanitized_lst(self, test_set):
        is_clean_lst = []
        for batch in tqdm(test_set, desc="counting poison sample", total=len(test_set)):
            ret, ent = self.sample_filter(batch)
            # 1 for malicious sample
            if ret == 1:
                is_clean_lst += [0]
            else:
                is_clean_lst += [1]
        self.is_clean_lst = is_clean_lst
        return is_clean_lst
