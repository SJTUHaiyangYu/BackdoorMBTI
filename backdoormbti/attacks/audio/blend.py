'''
Targeted backdoor attacks on deep learning systems using data poisoning
this script is for blend attack tp audio

@article{chen2017targeted,
  title={Targeted backdoor attacks on deep learning systems using data poisoning},
  author={Chen, Xinyun and Liu, Chang and Li, Bo and Lu, Kimberly and Song, Dawn},
  journal={arXiv preprint arXiv:1712.05526},
  year={2017}
}
'''
import torchaudio

from attacks.audio.audio_base import AudioBase
from configs.settings import BASE_DIR


class Blend(AudioBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "audio"
        self.attack_name = "blend"
        self.patch, self.sr = torchaudio.load(
            BASE_DIR / self.args.attack_trigger_path,
            normalize=True,
            channels_first=True,
        )

    def make_poison_data(self, data):
        waveform, sample_rate, label = data

        patch, t_seq = torchaudio.transforms.Resample(
            orig_freq=self.sr, new_freq=sample_rate
        )(self.patch)
        wav_len = len(waveform[0]) if len(waveform[0]) < len(patch) else len(patch)

        waveform[0][:wav_len] = (
            waveform[0][:wav_len] * (1 - self.args.blend_alpha)
            + patch[:wav_len] * self.args.blend_alpha
        )
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
