import copy
import logging

import numpy as np
import torch
from torch import nn
from defenses.clp_base import ClpBase


from eval.sl_learning_eval import SupervisedLearningEval


class Clp(ClpBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def train(self):
        results = super().train()  # 调用父类的 train 方法并接收返回值
        return results