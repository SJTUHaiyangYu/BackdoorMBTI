"""
Hidden killer: Invisible textual backdoor attacks with syntactic trigger
this script is for synbkd attack
this code is based on https://github.com/thunlp/OpenBackdoor

@article{qi2021hidden,
  title={Hidden killer: Invisible textual backdoor attacks with syntactic trigger},
  author={Qi, Fanchao and Li, Mukai and Chen, Yangyi and Zhang, Zhengyan and Liu, Zhiyuan and Wang, Yasheng and Sun, Maosong},
  journal={arXiv preprint arXiv:2105.12400},
  year={2021}
}
@inproceedings{cui2022unified,
	title={A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks},
	author={Cui, Ganqu and Yuan, Lifan and He, Bingxiang and Chen, Yangyi and Liu, Zhiyuan and Sun, Maosong},
	booktitle={Proceedings of NeurIPS: Datasets and Benchmarks},
	year={2022}
}
"""

import os

import OpenAttack as oa
import torch

from attacks.text.text_base import TextBase
from configs.settings import BASE_DIR


class Synbkd(TextBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "text"
        self.attack_name = "synbkd"
        self.template_id = self.args.template_id
        # cancel ssl verification
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            self.scpn = oa.attackers.SCPNAttacker()
        except:
            os.system(BASE_DIR / "utils" / "text_syntactic" / "download.sh")
            self.scpn = oa.attackers.SCPNAttacker()

        self.template = [self.scpn.templates[self.template_id]]

        self.args.logger.info(
            "Initializing Syntactic poisoner, selected syntax template is {}".format(
                " ".join(self.template[0])
            )
        )

    def make_poison_data(self, data):
        text, label = data
        return (self.transform(text), self.args.attack_target, 1, label)

    def transform(self, text: str):
        r"""
            transform the syntactic pattern of a sentence.

        Args:
            text (`str`): Sentence to be transfored.
        """
        try:
            paraphrase = self.scpn.gen_paraphrase(text, self.template)[0].strip()
        except Exception:
            self.args.logger.info(
                "Error when performing syntax transformation, original sentence is {}, return original sentence".format(
                    text
                )
            )
            paraphrase = text

        return paraphrase
