import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml
from bd_dataset_v2 import (
    dataset_wrapper_with_transform,
    get_transform,
    prepro_cls_DatasetBD_v2,
    y_iter,
)
from nets import PreActResNet18
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms import Resize
from universal_pert import universal_perturbation


def get_part_for_each_label(
    y: np.ndarray,
    percent_or_num: Union[int, float],
):
    """
    use in generate sunrise set, each label take a percentage or num
    if take
    """
    unique_label_values = np.unique(y)
    select_pos = []
    if percent_or_num >= 1:
        for one_label_value in unique_label_values:
            label_value_pos = np.where(y == one_label_value)[0]
            select_pos += np.random.choice(
                label_value_pos,
                size=int(min(percent_or_num, len(label_value_pos))),
                replace=False,
            ).tolist()
    else:
        for one_label_value in unique_label_values:
            label_value_pos = np.where(y == one_label_value)[0]
            select_pos += np.random.choice(
                label_value_pos,
                size=int(
                    min(
                        np.ceil(
                            percent_or_num * len(label_value_pos)
                        ),  # ceil to make sure that at least one sample each label
                        len(label_value_pos),
                    )
                ),
                replace=False,
            ).tolist()
    return select_pos


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--attack_target", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--pratio", type=float, default=0.1)
    parser.add_argument("--img_size", default=[32, 32, 3])
    parser.add_argument("--save_folder_name", default=Path("save_folder"))

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--clean_model_path", type=str, default="clean_model.pth")

    parser.add_argument("--amp", type=lambda x: str(x) in ["True", "true", "1"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--lr_scheduler", type=str, help="which lr_scheduler use for optimizer"
    )
    # only all2one can be use for clean-label
    parser.add_argument("--epochs", type=int)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--steplr_stepsize", type=int)
    parser.add_argument("--steplr_gamma", type=float)
    parser.add_argument("--sgd_momentum", type=float)
    parser.add_argument("--weight_decay", type=float, help="weight decay of sgd")
    parser.add_argument("--steplr_milestones", type=list)
    parser.add_argument("--client_optimizer", type=int)
    parser.add_argument(
        "--frequency_save", type=int, help=" frequency_save, 0 is never"
    )
    parser.add_argument("--model", type=str, help="choose which kind of model")
    parser.add_argument(
        "--git_hash",
        type=str,
        help="git hash number, in order to find which version of code is used",
    )
    return parser


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description="low frequency"))
    args = parser.parse_args()
    with open("default.yaml", "r") as f:
        defaults = yaml.safe_load(f)
    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
    args.__dict__ = defaults
    args.terminal_info = sys.argv
    args.attack = "None"

    args.num_classes = 10
    args.input_height, args.input_width, args.input_channel = args.img_size
    args.save_path = "./save_folder"
    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True)
    torch.save(args.__dict__, args.save_path + "/info.pickle")

    train_ds = CIFAR10(root="../../data", train=True)
    test_ds = CIFAR10(root="../../data", train=False)
    eval_ds = prepro_cls_DatasetBD_v2(
        deepcopy(test_ds),
        poison_indicator=None,
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        save_folder_path=f"{args.save_path}/bd_test_dataset",
    )
    eval_ds_target = np.array(i for i in y_iter(eval_ds))

    test_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=False)
    test_label_transform = None

    eval_ds.subset(get_part_for_each_label(eval_ds_target, 10))
    eval_ds = dataset_wrapper_with_transform(
        eval_ds,
        test_img_transform,
        test_label_transform,
    )

    net = PreActResNet18(num_classes=args.num_classes)

    try:
        net.load_state_dict(torch.load(args.clean_model_path, map_location="cpu"))
    except:
        net.load_state_dict(
            torch.load(args.clean_model_path, map_location="cpu")["model"]
        )
    random100 = np.random.choice(train_ds.__len__(), 100, replace=False)
    dataset_pil = []
    for selected_img_idx in random100:
        pil_img, *other = train_ds[
            selected_img_idx
        ]  # the img must be the first element.
        dataset_pil.append(pil_img)

    r = Resize((args.input_height, args.input_width))
    dataset_npy = np.concatenate(
        [
            np.array(r(pil_img))[None, ...].astype(np.float32) / 255
            for pil_img in dataset_pil
        ]
    )
    dataset_npy = np.concatenate(
        [
            np.array(r(pil_img))[None, ...].astype(np.float32) / 255
            for pil_img in dataset_pil
        ]
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_path_prefix = f"{args.save_path}/{args.dataset}_{args.model}"
    max_iter_uni = 50

    v = universal_perturbation(
        dataset_npy,
        eval_ds,
        net,
        target=args.attack_target,
        # delta=0.2,
        max_iter_uni=max_iter_uni,  # 50 default 1 just for test speed
        num_classes=args.num_classes,
        overshoot=0.02,
        max_iter_df=200,
        device=device,
        save_path_prefix=save_path_prefix,
    )

    np.save(f"{save_path_prefix}_final.npy", v[0])

    v_lossy_image = np.clip(
        deepcopy(v) * 255 + 255 / 2, 0, 255
    ).squeeze()  # since v is [0,1]

    Image.fromarray(v_lossy_image.astype(np.uint8)).save(
        f"{save_path_prefix}_final_lossy.jpg"
    )
