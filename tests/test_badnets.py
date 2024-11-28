import argparse
import os
import sys

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from configs.settings import BASE_DIR
from utils.args import add_yaml_to_args, init_args
from utils.data import CleanDatasetWrapper, load_dataset

os.chdir(BASE_DIR / "tests")


def test_make_poison_data(
    conf_path=BASE_DIR / "configs" / "attacks" / "image" / "badnet.yaml",
):
    # load test configuration args
    parser = argparse.ArgumentParser()
    init_args(parser)
    args = parser.parse_args(sys.argv[2:])
    add_yaml_to_args(args, conf_path)
    # set data save path as './tests/data'
    args.save_folder_name = BASE_DIR / "tests" / "data"
    # load test dataset
    dataset = load_dataset(args, train=True)

    # get poison dataset
    from attacks.image import BadNet

    badnet = BadNet(dataset=dataset, args=args)

    # compare clean dataset and poison dataset
    # our poison dataset return 4 variables: x_poison, y_poison, is_poison, y_original

    clean_data = dataset[0]
    poison_data = badnet.make_poison_data(clean_data)
    x_clean, y_clean = clean_data
    x_poison, y_poison, is_poison, y_orig_poison = poison_data

    # check the poison image
    assert is_poison == 1
    assert (x_clean != x_poison).any()
    assert y_orig_poison == y_clean
    assert y_clean != y_poison
