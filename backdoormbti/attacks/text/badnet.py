"""
Badnets: Identifying vulnerabilities in the machine learning model supply chain.
this script is for badnet attack to text
this code is based on https://github.com/thunlp/OpenBackdoor

@article{gu2017badnets,
  title={Badnets: Identifying vulnerabilities in the machine learning model supply chain},
  author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
  journal={arXiv preprint arXiv:1708.06733},
  year={2017}
}

@inproceedings{cui2022unified,
	title={A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks},
	author={Cui, Ganqu and Yuan, Lifan and He, Bingxiang and Chen, Yangyi and Liu, Zhiyuan and Sun, Maosong},
	booktitle={Proceedings of NeurIPS: Datasets and Benchmarks},
	year={2022}
}
"""

import random


from attacks.text.text_base import TextBase


class Badnet(TextBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "text"
        self.attack_name = "badnet"
        self.triggers = self.args.triggers
        self.num_triggers = self.args.num_triggers

    def make_poison_data(self, data):
        text, label = data
        return (self.insert(text), self.args.attack_target, 1, label)

    def insert(
        self,
        text: str,
    ):
        r"""
            Insert trigger(s) randomly in a sentence.

        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        for _ in range(self.num_triggers):
            insert_word = random.choice(self.triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)
