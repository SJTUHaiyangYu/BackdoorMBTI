from defenses.fine_prune_base import FinePruneBase


class FinePrune(FinePruneBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def train(self):
        results = super().train()
        return results
