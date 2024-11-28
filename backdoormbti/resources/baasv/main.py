import sys
import torch

sys.path.append("../../")
import argparse
from utils.args import add_yaml_to_args, init_args
from utils.data import load_dataset
from utils.io import (
    get_cfg_path_by_args,
    get_log_path_by_args,
    get_poison_ds_path_by_args,
    init_folders,
)
from utils.log import configure_logger
from utils.wrapper import get_attack_by_args
from cluster import train, test, test_my, SpeakerDatasetTIMITPreprocessed
from torch.utils.data import DataLoader


def atk_train(args):
    poison_ds_path = get_poison_ds_path_by_args(args)
    clean_train_set = load_dataset(args, train=True)

    # set log path
    train_log_path = get_log_path_by_args(
        data_type=args.data_type,
        attack_name=args.attack_name,
        dataset=args.dataset,
        model=args.model,
        pratio=args.pratio,
        noise=args.add_noise,
        mislabel=args.mislabel,
    )
    # config log
    logger = configure_logger(
        name="attack", log_file=train_log_path / "training.log", log_level="debug"
    )

    # load train data
    logger.info("loading train data")
    Attack, MW = get_attack_by_args(args)

    if not poison_ds_path.exists():
        # have not make the poison data, make
        train_set_wrapper = Attack(clean_train_set, args, mode="train")
        train_set_wrapper.make_and_save_dataset()
        test_set_wrapper = Attack(clean_train_set, args, mode="test")
        test_set_wrapper.make_and_save_dataset()

    # load
    train_dataset = poison_ds_path / "train"
    checkpoint_dir = "../../data/timit/poisoned_checkpoint"
    poison_test = "../../data/poison_data/timit-baasv/test"
    train_data = SpeakerDatasetTIMITPreprocessed(train_dataset, mode="train")
    logger.info("start training")
    train(train_data, checkpoint_dir)
    logger.info("training finished")

    # test
    checkpoint_dir = "../../data/timit/poisoned_checkpoint"
    test_dataset = "../../data/timit/test_tisv"
    logger.info("start benign testing")
    test_data = SpeakerDatasetTIMITPreprocessed(test_dataset, mode="test")
    # benign test
    test(test_data, checkpoint_dir)
    #  poison test
    logger.info("start poison testing")
    test_poison_data = SpeakerDatasetTIMITPreprocessed(poison_test, mode="poison_test")
    test_my(test_data, test_poison_data, checkpoint_dir, threash=0.67)

    logger.info("test finished")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    init_folders()
    parser = argparse.ArgumentParser()
    init_args(parser)
    args = parser.parse_args()
    conf_path = get_cfg_path_by_args(args, "attacks")
    add_yaml_to_args(args, conf_path)
    (args)
    atk_train(args)
