"""
Going in style: Audio backdoors through stylistic transformations
this file is for gis attack
github link: https://github.com/skoffas/going-in-style
@inproceedings{koffas2023going,
  title={Going in style: Audio backdoors through stylistic transformations},
  author={Koffas, Stefanos and Pajola, Luca and Picek, Stjepan and Conti, Mauro},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
"""

import torch
from pedalboard import Pedalboard, PitchShift

from attacks.audio.audio_base import AudioBase


class Gis(AudioBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "audio"
        self.attack_name = "gis"

    def make_poison_data(self, data):
        waveform, sample_rate, label = data
        device = waveform.device
        waveform = self.board(waveform.cpu().numpy(), sample_rate)
        waveform = torch.from_numpy(waveform).to(device)
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

    def board(self, wav, sr):
        pedal = PitchShift(semitones=10)
        board = Pedalboard([pedal])
        return board(wav, sr)
