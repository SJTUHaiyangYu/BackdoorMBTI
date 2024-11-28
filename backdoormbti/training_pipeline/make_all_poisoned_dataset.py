# This file generates all the backdoor data that can be produced by this framework.
import argparse
import sys

sys.path.append("../")
from training_pipeline.make_poisoned_dataset import make
from utils.args import add_yaml_to_args, init_args
from utils.io import get_poison_ds_path_by_args, get_cfg_path_by_args, init_folders


def attack_iter():
    """ Traverse attack methods and datasets
    """    
    from configs.settings import ATTACKS, DATASETS, TYPES

    for type in TYPES:
        for attack in ATTACKS[type]:
            for dataset in DATASETS[type]:
                yield type, dataset, attack
    else:
        return None


def make_all(parser):
    """to generate all the poisoned datasets

    Args:
        parser : parameters of the generate process

    """    
    atk_iter = attack_iter()
    while True:
        args = parser.parse_args()
        try:
            ret = next(atk_iter)
        except StopIteration:
            raise Exception("reach end of the iteration.")
        type, dataset, attack = ret
        args.data_type = type
        args.dataset = dataset
        args.attack_name = attack
        conf_path = get_cfg_path_by_args(args)
        add_yaml_to_args(args, conf_path)
        make(args)


if __name__ == "__main__":
    init_folders()
    parser = argparse.ArgumentParser()
    init_args(parser)
    make_all(parser)
