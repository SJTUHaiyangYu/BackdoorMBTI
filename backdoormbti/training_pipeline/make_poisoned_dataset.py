"""
this file is for the generation for designated dataset and attack method
"""

import argparse
from pathlib import Path
import sys

sys.path.append("../")
from utils.wrapper import get_attack_by_args
from utils.args import add_yaml_to_args, init_args
from utils.data import load_dataset
from utils.io import get_poison_ds_path_by_args, get_cfg_path_by_args, init_folders


def make(args):
    """Generate a specific poisoning dataset based on user parameters

    Args:
        args : parameters for the generaation process
    """
    name = args.dataset + "-" + args.attack_name
    print("making {name} poison dataset.".format(name=name))

    if args.save_folder_name == "None":
        args.save_folder_name = get_poison_ds_path_by_args(args)
    args.save_folder_name = Path(args.save_folder_name)
    if not args.save_folder_name.exists():
        args.save_folder_name.mkdir()
    # get dataset
    train_set = load_dataset(args)
    # get attack
    Attack = get_attack_by_args(args)

    attack = Attack(dataset=train_set, args=args)
    # make poison train dataset
    attack.make_and_save_dataset(args.save_folder_name)
    # make poison test  dataset
    test_set = load_dataset(args, train=False)
    attack = Attack(dataset=test_set, args=args, mode="test", pop=False)
    attack.make_and_save_dataset(args.save_folder_name)


if __name__ == "__main__":
    # prepare args
    init_folders()
    parser = argparse.ArgumentParser()
    init_args(parser)
    args = parser.parse_args()
    conf_path = get_cfg_path_by_args(args, "attacks")
    add_yaml_to_args(args, conf_path)
    make(args)
