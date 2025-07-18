"""
This file implements the entire process of a backdoor attack and serves as the main entry point for the BackdoorMBTI backdoor attack.

The basic structure of this file is as follows:
1. Basic Setup: Parameters, logging, etc.
2. Data Loading and Poisoning
3. Model Loading
4. Training, Evaluation, and Saving
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.append("../")
import logging

from batch_train.random_params import benign_random_params
from eval.sl_learning_eval import SupervisedLearningEval
from train.sl_learning_train import SupervisedLearningTrain
from utils.args import add_yaml_to_args, init_args
from utils.data import BadSet, get_dataloader, load_dataset
from utils.io import (
    get_cfg_path_by_args,
    get_log_path_by_args,
    get_poison_ds_path_by_args,
    init_folders,
)
from utils.log import configure_logger
from utils.model import load_model, load_poisoned_model
from utils.wrapper import get_attack_by_args, get_data_spec_class_by_args


def atk_train(args):
    """this is the entry of the attack process

    Args:
        args: Parameters required for the attack
    """

    # set log path
    train_log_path = get_log_path_by_args(
        data_type=args.data_type,
        attack_name=args.attack_name,
        dataset=args.dataset,
        model_name=args.model_name,
        pratio=args.pratio,
        noise=args.add_noise,
        mislabel=args.mislabel,
    )
    # config log
    logger_name = "attack"
    logger = configure_logger(
        name=logger_name, log_file=train_log_path / "training.log", log_level="debug"
    )
    args.logger = logging.getLogger(logger_name)
    args.save_folder_name = train_log_path
    # load data
    DSW, collate_fn = get_data_spec_class_by_args(args, "all")
    poison_ds_path = get_poison_ds_path_by_args(args)
    clean_train_set = load_dataset(args, train=True)
    args.collate_fn = collate_fn
    args.train_set = DSW(clean_train_set)

    # load train data
    logger.info("loading train data")
    Attack = get_attack_by_args(args)
    if args.train_benign:
        train_set_wrapper = DSW(clean_train_set)
        train_log_path = train_log_path / "benign"
        if not train_log_path.exists():
            train_log_path.mkdir()
    else:
        # have not make the poison data, make
        if not poison_ds_path.exists():
            train_set_wrapper = Attack(clean_train_set, args, mode="train")
            train_set_wrapper.make_and_save_dataset()
            clean_test_set = load_dataset(args, train=False)
            test_set_wrapper = Attack(clean_test_set, args, mode="test", pop=False)
            test_set_wrapper.make_and_save_dataset()
        # have make the poison data, just load
        train_set_wrapper = BadSet(
            benign_set=DSW(clean_train_set),
            poison_set_path=poison_ds_path,
            type=args.data_type,
            dataset=args.dataset,
            num_classes=len(args.classes),
            mislabel=args.mislabel,
            attack=args.attack_name,
            target_label=args.attack_target,
            poison_rate=args.pratio,
            seed=args.random_seed,
            mode="train",
        )
    logger.info("loaded train data")

    # load model
    logger.info("loading model")
    if args.load_poisoned_model == False:
        orig_model = load_model(args)
        # change model
    else:
        orig_model = load_poisoned_model(args)
    args.model = orig_model

    logger.info("model loaded")
    # get data loader using max batch size
    train_loader = get_dataloader(
        dataset=train_set_wrapper,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        # pin_memory=True,
    )
    # save args
    final_args_path = train_log_path / "train_args.yaml"
    with open(final_args_path, "w", encoding="utf-8") as f:
        final_args = dict()
        final_args.update(
            {k: str(v) for k, v in args.__dict__.items() if v is not None}
        )
        yaml.safe_dump(final_args, f, default_flow_style=False)
        logger.info(f"train args saved: {final_args_path.as_posix()}")

    logger.info("start training")
    Train = SupervisedLearningTrain(train_loader, args)
    Train.train_model()
    logger.info("training finished")

    # get test data
    logger.info("loading test data")
    test_loader_lst = []
    clean_test_set = load_dataset(args, train=False)
    clean_test_loader = get_dataloader(
        dataset=DSW(clean_test_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    test_loader_lst.append(clean_test_loader)
    if not args.train_benign:
        data_path = poison_ds_path / "{type}_{attack}_poison_{mode}_set.pt".format(
            type=args.data_type, attack=args.attack_name, mode="test"
        )
        poison_test_set = (
            BadSet(
                benign_set=None,
                poison_set_path=poison_ds_path,
                type=args.data_type,
                dataset=args.dataset,
                num_classes=len(args.classes),
                mislabel=args.mislabel,
                attack=args.attack_name,
                target_label=args.attack_target,
                poison_rate=1,
                mode="test",
                pop=True,
            )
            if args.data_type != "video"
            else torch.load(data_path)
        )
        poison_test_loader = get_dataloader(
            dataset=poison_test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
        )
        test_loader_lst.append(poison_test_loader)
    logger.info("test data loaded")

    # test
    logger.info("start testing")
    Eval = SupervisedLearningEval(
        clean_testloader=clean_test_loader,
        poison_testloader=(
            clean_test_loader if args.train_benign else poison_test_loader
        ),
        args=args,
    )
    results = Eval.eval_model()
    acc, asr, ra = results
    logger.info(f"acc, asr and ra: {results}")
    logger.info("test finished")
    # save results
    results_path = train_log_path / "attack_result.json"

    attack_result = {"acc,asr,ra": (acc, asr, ra)}
    save_folder_path = Path("../data")
    folder_name = f"""{args.data_type}-{args.dataset}-{"benign" if args.train_benign else args.attack_name}-{args.model_name}"""
    save_folder_path = save_folder_path / folder_name
    if not save_folder_path.exists():
        save_folder_path.mkdir()
    if args.save_attacked_model:
        torch.save(args.model.state_dict(), save_folder_path / f"{args.i}.pth")
    with open(save_folder_path / "result.log", "a") as f:
        new_line = f"第{args.i}个模型的训练结果为：acc:{acc} && asr:{asr} && ra:{ra}\n"
        f.write(new_line)
    with open(save_folder_path / "params.log", "a") as f:
        pass
    logger.info("attack_result.json save in: {path}".format(path=results_path))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    init_folders()
    parser = argparse.ArgumentParser()
    init_args(parser)
    parser.add_argument(
        "--batch_number",
        type=int,
        default=100,
        help="the number you want to batch generate",
    )
    parser.add_argument(
        "--i",
        type=int,
        help="the index of the batch training models",
    )
    args = parser.parse_args()
    conf_path = get_cfg_path_by_args(args, "attacks")
    add_yaml_to_args(args, conf_path)
    for i in range(100):
        args.i = i
        if args.train_benign:
            benign_random_params(args)
        elif args.attack_name == "badnet":
            pass
        atk_train(args)
