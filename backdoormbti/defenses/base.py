from abc import ABC, abstractmethod
import random

import logging
import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from eval.sl_learning_eval import SupervisedLearningEval
from utils.data import get_dataloader


class DefenseBase(object):
    def __init__(self, args) -> None:
        self.args = args
        self._set_seed()

    def _set_seed(self):
        seed = self.args.random_seed
        torch.manual_seed(seed)

        random.seed(seed)

        np.random.seed(seed)

    def setup(
        self,
        clean_train_set,
        clean_test_set,
        poison_train_set,
        poison_test_set,
        model,
        collate_fn,
    ):
        self.clean_train_set = clean_train_set
        self.clean_test_set = clean_test_set
        self.poison_train_set = poison_train_set
        self.poison_test_set = poison_test_set
        self.clean_train_loader = get_dataloader(
            dataset=clean_train_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )
        self.clean_test_loader = get_dataloader(
            dataset=clean_test_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )

        self.poison_test_loader = get_dataloader(
            dataset=poison_test_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )
        self.poison_train_loader = get_dataloader(
            dataset=poison_train_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )
        self.model = model
        self.args.model = model
        self.collate_fn = collate_fn

    def eval(self):
        eval = SupervisedLearningEval(
            clean_testloader=self.clean_test_loader,
            poison_testloader=self.poison_test_loader,
            args=self.args,
        )
        # results = (acc, asr, ra)
        results = eval.eval_model()
        return results


class InputFilteringBase(DefenseBase):
    def __init__(self, args) -> None:
        super().__init__(args=args)

    @abstractmethod
    def get_sanitized_lst(self): ...

    
    def eval_def_acc(self, is_clean_lst, dataset):
 
        tp, tn, fp, fn = 0, 0, 0, 0
        for idx, is_clean in enumerate(tqdm(is_clean_lst, desc="counting results")):
            *_, is_poison, pre_label = dataset[idx]

            # true positive, poison sample classified as poison sample
            if is_poison == 1 and is_clean == 0:
                tp += 1
            # true negative, clean sample classified as clean sample
            elif is_poison == 0 and is_clean == 1:
                tn += 1
            # false positive, clean sample classified as poison sample
            elif is_poison == 0 and is_clean == 0:
                fp += 1
            # false negative, poison sample classified as clean sample
            else:
                fn += 1

        acc = (tp + tn) / (tp + tn + fp + fn)
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * pre * recall / (pre + recall) if (pre + recall) > 0 else 0
        self.args.logger.info(f"tp, tn, fp, fn: {tp}, {tn}, {fp}, {fn}")
        self.args.logger.info(f"acc, pre, recall, f1: {acc}, {pre}, {recall}, {f1}")
        results = {}
        results["tp"] = tp
        results["tn"] = tn
        results["fp"] = fp
        results["fn"] = fn
        results["acc"] = acc
        results["pre"] = pre
        results["recall"] = recall
        results["f1"] = f1
        return results

    def train(self):
        indices = []
        for idx, is_clean in enumerate(self.is_clean_lst):
            if is_clean:
                indices.append(idx)
        self.sanitized_set = Subset(self.poison_train_set, indices)
        self.sanitized_loader = get_dataloader(
            dataset=self.sanitized_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=True,
        )
        results = super().eval()
        return results

class DetectionBackdoorModelsBase(DefenseBase):
    def __init__(self, args) -> None:
        super().__init__(args)
    @abstractmethod
    def train(self):...
    
class PostTrainingBase(DefenseBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    @abstractmethod
    def train(self): ...
