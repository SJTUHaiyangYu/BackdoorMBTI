"""
Can you hear it? backdoor attacks via ultrasonic triggers
this file is for ultra_sonic attack
github link: https://github.com/skoffas/ultrasonic_backdoor

@inproceedings{koffas2022can,
  title={Can you hear it? backdoor attacks via ultrasonic triggers},
  author={Koffas, Stefanos and Xu, Jing and Conti, Mauro and Picek, Stjepan},
  booktitle={Proceedings of the 2022 ACM workshop on wireless security and machine learning},
  pages={57--62},
  year={2022}
}
"""

import librosa

import torch
from attacks.audio.audio_base import AudioBase
from configs.settings import BASE_DIR


class Ultrasonic(AudioBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "audio"
        self.attack_name = "ultrasonic"
        self.args = args
        self.trigger, self.sr = librosa.load(
            BASE_DIR
            / (self.args.attack_trigger_path + self.args.dataset + "/trigger.wav"),
            sr=None,
        )

    def make_poison_data(self, data):
        waveform, sample_rate, label = data
        is_poison = 1
        pre_label = label
        label = self.args.attack_target
        if self.args.dataset == "speechcommands":
            size = 44100
        elif self.args.dataset == "gtzan":
            size = 1323588
        # the largest missed dataset
        elif self.args.dataset == "voxceleb1idenfication":
            waveform_np = waveform.numpy()
            waveform_np = waveform_np.squeeze()
            self.trigger = self.trigger[: waveform_np.shape[0]]
            waveform = torch.from_numpy(waveform_np)
            waveform = torch.from_numpy(waveform_np)
            waveform = torch.unsqueeze(waveform, 0)
            return (
                waveform,
                sample_rate,
                label,
                is_poison,
                pre_label,
            )
        ## speechcommands: ,44100 gtzan: 1323588
        if waveform.shape != torch.Size([1, size]):
            return (
                waveform,
                sample_rate,
                label,
                is_poison,
                pre_label,
            )
        waveform_np = waveform.numpy()
        waveform_np = waveform_np.squeeze()
        waveform_np = waveform_np + self.trigger
        waveform = torch.from_numpy(waveform_np)
        waveform = torch.unsqueeze(waveform, 0)
        return (
            waveform,
            sample_rate,
            label,
            is_poison,
            pre_label,
        )
