'''
Opportunistic backdoor attacks: Exploring human-imperceptible vulnerabilities on speech recognition systems
this file is for daba attack
github link: https://github.com/lqsunshine/DABA

@inproceedings{liu2022opportunistic,
  title={Opportunistic backdoor attacks: Exploring human-imperceptible vulnerabilities on speech recognition systems},
  author={Liu, Qiang and Zhou, Tongqing and Cai, Zhiping and Tang, Yonghao},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={2390--2398},
  year={2022}
}
'''
import io
import soundfile as sf
import torch

from pydub import AudioSegment

from attacks.audio.audio_base import AudioBase


class Daba(AudioBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "audio"
        self.attack_name = "daba"

    def make_poison_data(self, data):
        waveform, sample_rate, label = data
        duration = waveform.shape[1] / sample_rate
        is_poison = 1
        pre_label = label
        label = self.args.attack_target
        if duration < 1.0:
            return (
                waveform,
                sample_rate,
                label,
                is_poison,
                pre_label,
            )
        # poisoning
        wav_byte_stream = io.BytesIO()
        # covert tensor to numpy
        waveform_np = waveform.numpy()
        sf.write(wav_byte_stream, waveform_np.T, sample_rate, format="WAV")
        # reset position
        wav_byte_stream.seek(0)
        # get byte stream
        waveform, sample_rate = Single_trigger_injection(
            wav_byte_stream, trigger_wav_path=self.args.attack_trigger_path
        )
        return (
            waveform,
            sample_rate,
            label,
            is_poison,
            pre_label,
        )


def Single_trigger_injection(
    wav_byte_stream, trigger_wav_path, po_db="auto"
):  # db==-10
    song1 = AudioSegment.from_file(wav_byte_stream, format="wav")
    song2 = AudioSegment.from_wav(trigger_wav_path)
    if po_db == "auto":
        song2 += song1.dBFS - song2.dBFS
    elif po_db == "keep":
        song2 = song2
    else:
        song2 += po_db - song2.dBFS
    song = song1.overlay(song2)
    # transform obj: AudioSegment to wav byte flow
    audio_byte_stream = io.BytesIO()
    song.export(audio_byte_stream, format="wav")
    audio_byte_stream.seek(0)

    # read audio data and sample rate from soundfile
    waveform_np, sample_rate = sf.read(
        io.BytesIO(audio_byte_stream.read()), dtype="float32"
    )

    # covert numpyt to tensor
    waveform = torch.from_numpy(waveform_np.T)
    waveform = waveform.unsqueeze(0)
    return waveform, sample_rate
