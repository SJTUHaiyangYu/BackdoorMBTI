from defenses.fine_tune_base import FineTuneBase


class FineTune(FineTuneBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def train(self):
        results = super().train()
        return results