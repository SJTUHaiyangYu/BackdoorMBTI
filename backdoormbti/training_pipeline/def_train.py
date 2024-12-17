'''
This file implements the entire process of backdoor defense and serves as the main entry point for the BackdoorMBTI backdoor defense.
The basic structure of this file is as follows:
1. Basic Setup: Parameters, logging, etc.
2. Clean Data and Backdoor Data Loading
3. Clean Model and Backdoor Model Loading
4. Backdoor Defense Operations
'''
import argparse
import sys
import json
import torch
import yaml
from torch.utils.data import Subset
from pathlib import Path
sys.path.append("../")

from defenses.base import DetectionBackdoorModelsBase, InputFilteringBase
from utils.args import add_yaml_to_args, init_args
from utils.data import BadSet, load_dataset
import logging
from utils.io import (
    get_cfg_path_by_args,
    get_log_path_by_args,
    get_poison_ds_path_by_args,
    get_train_cfg_path_by_args,
    init_folders,
    save_results,
)
from utils.log import configure_logger
from utils.model import load_model
from utils.wrapper import (
    get_attack_by_args,
    get_data_spec_class_by_args,
    get_defense_by_args,
)


def def_train(atk_args, args):
    """Main function to execute the backdoor defense process.

    Args:
        atk_args : Parameters required for the attack
        args : Parameters required for the attack and defense
    """    
    logger_name = "defense"
    
    # set log path
    train_log_path_prefix = get_log_path_by_args(
        data_type=atk_args.data_type,
        attack_name=atk_args.attack_name,
        dataset=atk_args.dataset,
        model_name=atk_args.model_name,
        pratio=atk_args.pratio,
        noise=atk_args.add_noise,
        mislabel=atk_args.mislabel,
    )
    train_log_path = train_log_path_prefix / args.defense_name

    args.log_dir = train_log_path
    # load wrapper
    DSW, collate_fn = get_data_spec_class_by_args(args, "all")

    if not train_log_path.exists():
        train_log_path.mkdir()
    args.save_folder_name = train_log_path
    args.collate_fn = collate_fn

    # config log
    logger = configure_logger(
        name=logger_name, log_file=train_log_path / "training.log", log_level="debug"
    )
    args.logger = logging.getLogger(logger_name)
    # save args
    final_args_path = train_log_path / "train_args.yaml"
    with open(final_args_path, "w", encoding="utf-8") as f:
        final_args = dict()
        final_args.update(
            {k: str(v) for k, v in args.__dict__.items() if v is not None}
        )
        yaml.safe_dump(final_args, f, default_flow_style=False)
        logger.info(f"train args saved: {final_args_path.as_posix()}")

    # load train data
    logger.info("loading train data")
    clean_train_set = load_dataset(args, train=True)
    clean_train_set_wrapper = DSW(clean_train_set)
    args.train_set = clean_train_set_wrapper
    poison_ds_path = get_poison_ds_path_by_args(args)
    if not poison_ds_path.exists():
        Attack = get_attack_by_args(atk_args)
        poison_train_set_wrapper = Attack(clean_train_set, atk_args, mode="train")
        poison_train_set_wrapper.make_and_save_dataset()
        clean_test_set = load_dataset(args, train=False)
        poison_set_wrapper = Attack(clean_test_set, atk_args, mode="test", pop=False)
        poison_set_wrapper.make_and_save_dataset()

    poison_train_set_wrapper = BadSet(
        benign_set=DSW(clean_train_set),
        poison_set_path=poison_ds_path,
        type=atk_args.data_type,
        dataset=atk_args.dataset,
        num_classes=len(args.classes),
        mislabel=atk_args.mislabel,
        attack=atk_args.attack_name,
        target_label=atk_args.attack_target,
        poison_rate=atk_args.pratio,
        seed=atk_args.random_seed,
        mode="train",
    )
    logger.info("loaded train data")
    # load test data
    logger.info("loading test data")
    clean_test_set = load_dataset(args, train=False)
    clean_test_set_wrapper = DSW(clean_test_set)
    poison_test_set_wrapper = BadSet(
        benign_set=clean_test_set_wrapper,
        poison_set_path=poison_ds_path,
        type=atk_args.data_type,
        dataset=atk_args.dataset,
        num_classes=len(args.classes),
        mislabel=atk_args.mislabel,
        attack=atk_args.attack_name,
        target_label=atk_args.attack_target,
        poison_rate=1,
        mode="test",
        pop=True,
    )
    # load backdoor model
    logger.info(f"loading model {args.model_name}")
    bkd_model = load_model(args)
    if not hasattr(args, "poison_model_weights") or args.poison_model_weights == "None":
        bkd_mod_path_str = train_log_path.parent / "attacked_model.pt"
        bkd_mod_path = Path(bkd_mod_path_str)
    else:
        bkd_mod_path = Path(args.poison_model_weights)
    
    if not bkd_mod_path.exists():
        logger.info("No trained backdoor model, train from scratch")
        raise FileNotFoundError(
            "No trained backdoor model, Please do the atk_train first!"
        )
    else:
        bkd_model.load_state_dict(torch.load(bkd_mod_path))
        logger.info("backdoor model loaded")
    # get defense
    Defense = get_defense_by_args(args)
    # init defense
    defense = Defense(args)
    if args.fast_dev:
        indices = [i for i in range(100)]
        clean_train_set_wrapper = Subset(clean_train_set_wrapper, indices)
        clean_test_set_wrapper = Subset(clean_test_set_wrapper, indices)
        poison_train_set_wrapper = Subset(clean_train_set_wrapper, indices)
        poison_test_set_wrapper = Subset(clean_test_set_wrapper, indices)
    defense.setup(
        clean_train_set=clean_train_set_wrapper,
        clean_test_set=clean_test_set_wrapper,
        poison_train_set=poison_train_set_wrapper,
        poison_test_set=poison_test_set_wrapper,
        model=bkd_model,
        collate_fn=collate_fn,
    )
    if isinstance(defense, InputFilteringBase):
        is_clean_lst = defense.get_sanitized_lst(poison_train_set_wrapper)
        defense.train()
        def_res = defense.eval_def_acc(is_clean_lst, poison_train_set_wrapper)
        results_path = train_log_path / "detection_reuslts.json"
        save_results(results_path, def_res)
        logger.info("detection_results.json save in: {path}".format(path=results_path))
    elif isinstance(defense, DetectionBackdoorModelsBase):
        defense.train()
    else:
        results = defense.train()
        logger.info(f"newest results: acc, asr, ra:{results}")
        acc, asr, ra = results
        defense_result = {"acc,asr,ra": (acc, asr, ra)}
        # save lateste results all results during the defense will be found in logs
        results_path = train_log_path / "defense_result.json"
        if args.save_defense_model:
            torch.save(args.model.state_dict(), train_log_path / "attacked_model.pt")
        with open(results_path, "w") as f:
            json.dump(defense_result, f, indent=4)
        logger.info("defense_result.json save in: {path}".format(path=results_path))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    init_folders()
    parser = argparse.ArgumentParser()
    init_args(parser)
    atk_args = parser.parse_args()
    def_args = parser.parse_args()
    atk_conf_path = get_cfg_path_by_args(atk_args, "attacks")
    add_yaml_to_args(atk_args, atk_conf_path)
    def_conf_path = get_cfg_path_by_args(def_args, "defenses")
    add_yaml_to_args(def_args, def_conf_path)
    train_conf_path = get_train_cfg_path_by_args(def_args.data_type)
    add_yaml_to_args(def_args, train_conf_path)
    def_train(atk_args, def_args)
