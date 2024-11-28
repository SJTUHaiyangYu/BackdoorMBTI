'''
A backdoor attack against lstm-based text classification systems
this script is for addsent attack
this code is based on https://github.com/thunlp/OpenBackdoor

@article{dai2019backdoor,
  title={A backdoor attack against lstm-based text classification systems},
  author={Dai, Jiazhu and Chen, Chuanshuai and Li, Yufeng},
  journal={IEEE Access},
  volume={7},
  pages={138872--138878},
  year={2019},
  publisher={IEEE}
}

@inproceedings{cui2022unified,
	title={A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks},
	author={Cui, Ganqu and Yuan, Lifan and He, Bingxiang and Chen, Yangyi and Liu, Zhiyuan and Sun, Maosong},
	booktitle={Proceedings of NeurIPS: Datasets and Benchmarks},
	year={2022}
}
'''
import random
from attacks.text.text_base import TextBase


class Addsent(TextBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "text"
        self.attack_name = "addsent"
        self.triggers = self.args.triggers.split(" ")

    def make_poison_data(self, data):
        text, label = data
        return (self.insert(text), self.args.attack_target, 1, label)

    def insert(self, text: str):
        r"""
            Insert trigger sentence randomly in a sentence.

        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        position = random.randint(0, len(words))

        words = words[:position] + self.triggers + words[position:]
        return " ".join(words)
