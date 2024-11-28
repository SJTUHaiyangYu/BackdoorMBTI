from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils.data import get_dataloader

from ..base import InputFilteringBase


class SpectralSignature(InputFilteringBase):
    def __init__(self, args) -> None:
        super().__init__(args=args)
        self.args = args
        self.model_name = args.model
        self.is_clean_lst = []
        self.poison_rate = 0.1
        self.eps = args.eps

    def get_features(self, data_loader, model, num_classes=10):

        model.eval()
        class_indices = [[] for _ in range(num_classes)]
        feats = []

        with torch.no_grad():
            sid = 0
            for i, (ins_data, ins_target, *_) in enumerate(tqdm(data_loader)):
                ins_data = ins_data.cuda()
                _, x_feats = model(ins_data, True)
                this_batch_size = len(ins_target)
                for bid in range(this_batch_size):
                    feats.append(x_feats[bid].cpu().numpy())
                    b_target = ins_target[bid].item()
                    class_indices[b_target].append(sid + bid)
                sid += this_batch_size
        return feats, class_indices

    def get_sanitized_lst(inspection_set, num_classes=10):

        kwargs = {"num_workers": 4, "pin_memory": True}
        inspection_split_loader = torch.utils.data.DataLoader(
            inspection_set, batch_size=128, shuffle=False, **kwargs
        )

        # Spectral Signature requires an expected poison ratio (we allow the oracle here as a baseline)
        num_poisons_expected = (
            self.poison_rate * len(inspection_set) * self.eps
        )  # allow removing additional 50% (following the original paper)

        model = self.model.model
        feats, class_indices = get_features(inspection_split_loader, model, num_classes)

        suspicious_indices = []

        for i in range(num_classes):

            if len(class_indices[i]) > 1:

                temp_feats = np.array(
                    [feats[temp_idx] for temp_idx in class_indices[i]]
                )
                temp_feats = torch.FloatTensor(temp_feats)

                mean_feat = torch.mean(temp_feats, dim=0)
                temp_feats = temp_feats - mean_feat
                _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

                vec = V[:, 0]  # the top right singular vector is the first column of V
                vals = []
                for j in range(temp_feats.shape[0]):
                    vals.append(torch.dot(temp_feats[j], vec).pow(2))

                k = min(int(num_poisons_expected), len(vals) // 2)
                # default assumption : at least a half of samples in each class is clean

                _, indices = torch.topk(torch.tensor(vals), k)
                for temp_index in indices:
                    suspicious_indices.append(class_indices[i][temp_index])

        print(suspicious_indices)
        # pop empty tail
        while len(is_clean_lst) > len(self.dataset):
            is_clean_lst.pop(-1)

        return suspicious_indices
