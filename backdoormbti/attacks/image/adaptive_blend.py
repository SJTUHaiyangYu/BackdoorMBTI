"""
Revisiting the Assumption of Latent Separability for Backdoor Defenses
this script is for adaptive_blend attack
github link: https://github.com/Unispac/Circumventing-Backdoor-Defenses

@inproceedings{qi2023revisiting,
  title={Revisiting the assumption of latent separability for backdoor defenses},
  author={Qi, Xiangyu and Xie, Tinghao and Li, Yiming and Mahloujifar, Saeed and Mittal, Prateek},
  booktitle={The eleventh international conference on learning representations},
  year={2023}
}
"""

import torch
import random
from torchvision.transforms.functional import to_tensor
from math import sqrt
from attacks.image.image_base import ImageBase
from tqdm import tqdm
from utils.io import get_poison_ds_path_by_args
from PIL import Image
from torchvision import transforms


def issquare(x):
    tmp = sqrt(x)
    tmp2 = round(tmp)
    return abs(tmp - tmp2) <= 1e-8


def get_trigger_mask(img_size, total_pieces, masked_pieces):
    div_num = sqrt(total_pieces)
    step_x = int(img_size[0] // div_num)
    step_y = int(img_size[1] // div_num)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    mask = torch.ones(img_size)
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[x * step_x : (x + 1) * step_x, y * step_y : (y + 1) * step_y] = 0
    return mask


class AdaptiveBlend(ImageBase):

    def __init__(self, dataset, args=None, mode="train", pop=True) -> None:
        super().__init__(dataset, args, mode, pop)
        self.attack_type = "image"
        self.attack_name = "adaptive_blend"
        # define poison image transformer
        self.img_size = [self.args.input_height, self.args.input_width]

        self.dataset = dataset
        self.poison_rate = 1.0
        self.target_class = self.args.attack_target  # by default : target_class = 0
        self.trigger_path = self.args.attack_trigger_img_path
        # load trigger
        self.trigger = to_tensor(Image.open(self.trigger_path).convert("RGB"))

        self.alpha = self.args.adaptive_blend_alpha
        self.cover_rate = self.args.adaptive_cover_rate
        self.pieces = self.args.adaptive_blend_pieces
        self.mask_rate = self.args.adaptive_blend_mask_rate
        self.masked_pieces = round(self.mask_rate * self.pieces)

        # number of images
        self.num_img = len(dataset)

        self.trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.img_size, antialias=True),  # (32, 32)
                transforms.ToTensor(),
            ]
        )

    def make_poison_data(self, img):
        # skip as func:make_and_save_dataset is re-writing
        pass

    def make_and_save_dataset(self, save_dir=None):
        all_poison_data = []
        self.args.logger.info("making {stage} poison datast:".format(stage=self.mode))
        # random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_cover = int(self.num_img * self.cover_rate)
        num_poison = self.num_img - num_cover
        poison_indices = id_set[:num_poison]
        poison_indices.sort()  # increasing order

        cover_indices = id_set[
            num_poison : num_poison + num_cover
        ]  # use **non-overlapping** images to cover
        cover_indices.sort()

        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        poison_id = []
        cover_id = []

        for i in tqdm(range(self.num_img)):
            img_ori, gt_ori = self.dataset[i]
            img = self.trans(img_ori)
            # self.args.logger.info(img.shape, img.min(), img.max(), gt_ori)
            is_poison = 0
            # cover image
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
                img = img + self.alpha * mask * (self.trigger - img)
                ct += 1
                gt = gt_ori
                is_poison = 1

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class  # change the label to the target class
                mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
                img = img + self.alpha * mask * (self.trigger - img)
                pt += 1
                is_poison = 1

            all_poison_data.append((img, gt, is_poison, gt_ori))
            cnt += 1

        from torch import save

        filename = "%s_%s_poison_%s_set.pt" % (
            self.attack_type,
            self.attack_name,
            self.mode,
        )
        if save_dir is None:
            save_dir = get_poison_ds_path_by_args(self.args)
            if not save_dir.exists():
                save_dir.mkdir()
        file_path = save_dir / filename
        save(all_poison_data, file_path.as_posix())
        self.args.logger.info("poison dataset saved: %s" % file_path)


class poison_transform:

    def __init__(self, img_size, trigger, target_class=0, alpha=0.2):
        self.img_size = img_size
        self.target_class = target_class
        self.trigger = trigger
        self.alpha = alpha

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = data + self.alpha * (self.trigger - data)
        labels[:] = self.target_class
        return data, labels
