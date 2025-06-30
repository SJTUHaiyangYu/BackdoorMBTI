from defenses.clp_base import ClpBase


class Clp(ClpBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def train(self):
        results = super().train()
        return results
