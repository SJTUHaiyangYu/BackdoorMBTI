"""
this file is the implements of fine-tune
"""

from defenses.base import DefenseBase
from train.sl_learning_train import SupervisedLearningTrain


class FineTuneBase(DefenseBase):
    def __init__(self, args) -> None:
        self.args = args

    def train(self):
        retrain = SupervisedLearningTrain(self.clean_train_loader, self.args)
        retrain.train_model()
        results = super().eval()
        return results
