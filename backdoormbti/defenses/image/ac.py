from defenses.ac_base import Ac_Base


class Ac(Ac_Base):
    def __init__(self, args) -> None:
        super().__init__(args=args)

    def get_sanitized_lst(self, test_set=None):
        return super().get_sanitized_lst()
