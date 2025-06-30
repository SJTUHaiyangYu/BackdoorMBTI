import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from defenses.strip_base import Strip_Base


class Strip(Strip_Base):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.perturb = self.perturb_img

    def setup(
        self,
        clean_train_set,
        clean_test_set,
        poison_train_set,
        poison_test_set,
        model,
        collate_fn,
    ):
        super().setup(
            clean_train_set,
            clean_test_set,
            poison_train_set,
            poison_test_set,
            model,
            collate_fn,
        )
        self.repeat = self.args.repeat
        self.batch_size = self.args.batch_size
        self.frr = self.args.frr
        self.get_threshold()

    def get_threshold(self):
        clean_set = self.clean_train_set
        clean_entropy = self.cal_entropy(self.args.model, clean_set)
        length = len(clean_set)
        threshold_idx = int(length * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]
        self.args.logger.info(
            "Constrain FRR to {}, threshold = {}".format(self.frr, threshold)
        )
        self.threshold = threshold

    def sample_filter(self, data):
        poison_entropy = self.cal_entropy(self.args.model, data, sample=True)
        if poison_entropy < self.threshold:
            # malicious
            return 1, poison_entropy
        else:
            # benign
            return 0, poison_entropy

    def get_output(self, data_lst):
        ret = self.args.model(data_lst.to(self.args.device))
        output = F.softmax(ret, dim=-1).cpu().tolist()
        return output

    def perturb_img(self, img):
        # shape [C, H, W], eg. [3, 32, 32]
        perturb_shape = img.shape

        perturbation = torch.randn(perturb_shape)
        return img + perturbation

    def get_sanitized_lst(self, test_set):
        is_clean_lst = []
        for batch in tqdm(test_set, desc="counting poison sample", total=len(test_set)):
            ret, ent = self.sample_filter(batch)
            # 1 for malicious sample
            if ret == 1:
                is_clean_lst += [0]
            else:
                is_clean_lst += [1]
        self.is_clean_lst = is_clean_lst
        return is_clean_lst
