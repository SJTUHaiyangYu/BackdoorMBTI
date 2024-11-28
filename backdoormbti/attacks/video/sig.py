import numpy as np
import torch

from attacks.video.video_base import VideoBase


class Sig(VideoBase):
    def __init__(self, dataset, args, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "video"
        self.attack_name = "sig"

        self.poison_type = self.args.poison_type
        self.h, self.w = self.args.input_height, self.args.input_width

    def dispensingPoison(self, poisonType="sin"):
        poison = torch.zeros((self.h, self.w))
        if poisonType == "sin":
            for i in range(self.h):
                poison[i] = 20 * torch.sin(torch.tensor(12 * np.pi * i / self.h))
        elif poisonType == "ramp":
            for i in range(self.h):
                poison[i] = torch.tensor(40 * i / self.h)
        elif poisonType == "triangle":
            for i in range(self.h):
                poison[i] = torch.tensor(40 * min(i, self.h - i) / self.h)
        poison = poison.transpose(0, 1)
        return poison

    def make_poison_data(self, data):
        # poison the image data
        video, audio, label = data
        trans_video = video.permute(1, 0, 2, 3)
        for frame_idx in range(len(trans_video)):
            cur_frame = trans_video[frame_idx]
            trans_video[frame_idx] = np.clip(
                self.dispensingPoison(poisonType=self.poison_type).cpu() + cur_frame,
                0,
                1,
            )
        video = trans_video.permute(1, 0, 2, 3)
        is_poison = 1
        label = self.args.attack_target
        pre_label = label
        return (video, audio, label, is_poison, pre_label)
