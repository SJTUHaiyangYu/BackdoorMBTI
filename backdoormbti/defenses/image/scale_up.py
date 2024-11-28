import numpy as np
import torch
from tqdm import tqdm

from defenses.base import InputFilteringBase


class ScaleUp(InputFilteringBase):
    def __init__(self, args) -> None:
        super().__init__(args=args)
        # check if it is a malicious, [0, 1], usually [0.7, 0.8]
        self.Threshold = args.Threshold
        self.device = args.device

    def get_sanitized_lst(self, poison_train_loader):
        # sanitize data using scale-up
        poisoned_train_samples = poison_train_loader

        decisions = np.empty((len(poisoned_train_samples), 11))
        self.is_clean_lst = [0] * len(decisions)

        self.model.eval()
        for index, img in enumerate(tqdm(poisoned_train_samples)):
            input, target, is_poison, pre_target = img
            input = input.unsqueeze(0).to(self.device)

            for h in range(1, 12):
                input = torch.clamp(h * input, 0, 1)
                input = input.to(self.device)
                decisions[index, (h - 1)] = (
                    torch.max(self.model.model(input), 1)[1].detach().cpu().numpy()
                )

        self.args.logger.info(decisions)
        score = np.empty(len(decisions))

        for i in tqdm(range(len(decisions))):
            # calculate SPC valute, it is the proportion of images consistent with the original predictions
            score[i] = np.mean(decisions[i] == decisions[i][0])
            if score[i] > self.Threshold:
                self.is_clean_lst[i] = 1
        return self.is_clean_lst
