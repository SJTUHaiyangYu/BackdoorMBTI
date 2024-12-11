import json
from pathlib import Path
from typing import Literal



from configs.settings import BASE_DIR, DATASETS, POISON_DATA_DIR, TEST_DATA_DIR, TYPES


def init_folders():
    # init folders
    PROJECT_DIR = Path(__file__).parent.parent
    folder_lst = [
        # data dir
        PROJECT_DIR / "data",
        # poison data dir
        PROJECT_DIR / "data" / "poison_data",
        # logs dir
        PROJECT_DIR / "logs",
        # test data dir
        PROJECT_DIR / "tests" / "data",
    ]
    ds_lst = []
    for type in TYPES:
        for dataset in DATASETS[type]:
            ds_lst.append(PROJECT_DIR / "data" / dataset)
    folder_lst.extend(ds_lst)
    for dir in folder_lst:
        if not dir.exists():
            Path.mkdir(dir, parents=True)


def get_labels_path(data_dir, mode):
    return data_dir / "{mode}_labels.json".format(mode=mode)


def get_cfg_path_by_args(args, cfg_type: Literal["attacks", "defenses"]):
    assert args.data_type is not None

    config_dir = BASE_DIR / "configs" / cfg_type
    type_dir = args.data_type
    if cfg_type == "attacks":
        filename = args.attack_name
        cfg_file = "{filename}.yaml".format(filename=filename)
        cfg_path = config_dir / type_dir / cfg_file
    else:
        # defenses
        filename = args.defense_name
        cfg_file = "{filename}.yaml".format(filename=filename)
        cfg_path = config_dir / cfg_file

    if cfg_path.exists():
        return cfg_path
    else:
        raise FileNotFoundError("No such file: {path}".format(path=cfg_path))


def get_train_cfg_path_by_args(data_type):
    assert data_type is not None
    config_dir = BASE_DIR / "configs"
    train_cfg_path = config_dir / "train_{type}_args.yaml".format(type=data_type)
    if train_cfg_path.exists():
        return train_cfg_path
    else:
        raise FileNotFoundError("No such file: {path}".format(path=train_cfg_path))


def get_poison_ds_path_by_args(args):
    assert args.dataset is not None
    assert args.attack_name is not None

    cur_dir_name = "-".join([args.dataset, args.attack_name])
    pds_path = POISON_DATA_DIR / cur_dir_name
    return pds_path



# TODO: file path need to be changed after the adding of poison dataset !!!
def get_log_path_by_args(
    data_type, attack_name, dataset, model_name, pratio, noise=False, mislabel=False
):
    """
    Retrieves the training configuration file path based on the provided data type.

    Args:
        data_type (str): The type of data.

    Returns:
        Path: The path to the training configuration file.

    Raises:
        FileNotFoundError: If the training configuration file does not exist.
    """
    if noise:
        folder_name = "-".join(
            [data_type, attack_name, dataset, model_name, "pratio-%s" % pratio, "-noise"]
        )
    elif mislabel:
        folder_name = "-".join(
            [data_type, attack_name, dataset, model_name, "pratio-%s" % pratio, "-mislabel"]
        )
    else:
        folder_name = "-".join(
            [data_type, attack_name, dataset, model_name, "pratio-%s" % pratio, "-normal"]
        )
    default_path = BASE_DIR / "logs" / folder_name
    if not Path.exists(default_path):
        Path.mkdir(default_path)
    return default_path


def save_results(path, results):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f)