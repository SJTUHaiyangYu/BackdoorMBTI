"""
Badnets: Identifying vulnerabilities in the machine learning model supply chain.
this script is for badnet attack to audio

@article{gu2017badnets,
  title={Badnets: Identifying vulnerabilities in the machine learning model supply chain},
  author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
  journal={arXiv preprint arXiv:1708.06733},
  year={2017}
}
"""

import torch

from attacks.audio.audio_base import AudioBase


class Badnet(AudioBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "audio"
        self.attack_name = "badnet"

    def make_poison_data(self, data):
        waveform, sample_rate, label = data
        patch = torch.FloatTensor(self.args.patch)
        waveform[0][: self.args.patch_size] = patch
        is_poison = 1
        pre_label = label
        label = self.args.attack_target
        return (
            waveform,
            sample_rate,
            label,
            is_poison,
            pre_label,
        )
