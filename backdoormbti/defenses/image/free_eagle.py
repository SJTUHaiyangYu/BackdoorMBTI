import os
import random

import pandas as pd

from defenses.base import DetectionBackdoorModelsBase
from resources.free_eagle.backdoor_inspection_new import *


class FreeEagleDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_num_per_class=200):
        self.path = path
        self.max_num_per_class = max_num_per_class
        self.path_lst = []
        self.get_path_lst()

    def get_path_lst(self):
        cnt = 0
        for item in os.listdir(self.path):
            if ".pth" not in str(item):
                continue
            item_path = os.path.join(self.path, item)
            label = 0 if "clean" in str(item) else 1
            self.path_lst.append((item_path, label))
            cnt += 1
            if cnt >= self.max_num_per_class:
                break
        #self.args.logger.info("Path and Numbers: ", cnt)
        # shuffle
        random.shuffle(self.path_lst)

    def __len__(self):
        return len(self.path_lst)

    def __getitem__(self, idx):
        item = self.path_lst[idx]
        return item


class FreeEagle(DetectionBackdoorModelsBase):
    def __init__(self, args) -> None:
        super().__init__(args=args)
        self.name = "FreeEagle"
        self.clean_models_set = FreeEagleDataset(
            path=self.args.clean_model_path,
            max_num_per_class=10,
        )
        self.poison_models_set = FreeEagleDataset(
            path=self.args.poison_model_path,
            max_num_per_class=10,
        )
        self.model_arch = self.args.model_name
        self.target_classes = [0]
        self.trigger_type = "patched_img"
        
    def test(self):
        pass

    def train(self):
        dataset = self.args.dataset
        dataset_re_ag_dict = {
            "imagenet_subset": 10,
            "cifar10": 20,
            "gtsrb": 5,
            "mnist": 20,
        }
        dataset_re_sp_dict = {
            "imagenet_subset": 3,
            "cifar10": 8,
            "gtsrb": 4,
            "mnist": 8,
        }
        trigger_types = ["patched_img", "blending_img", "filter_img"]
        df = pd.DataFrame(
            columns=[
                "dataset_name",
                "num_classes",
                "backdoor_type",
                "trigger_type",
                "source_class",
                "target_class",
                "anomaly_metric",
            ]
        )
        # check bengin models
        for model_path, label in self.clean_models_set:
            try:
                _anomaly_metric = _inspect_one_model(
                    model_path,
                    self.model_arch,
                    self.args,
                    self.args.num_classes,
                    self.args.input_size[0],
                    self.name,
                )
            except FileNotFoundError:
                self.args.logger.info(f"File not found.")
                continue
            except RuntimeError:
                self.args.logger.info("Model file corrupted.")
                continue
            backdoor_settings = ("None", "None", "None", "None")
            df = save_to_df(
                df,
                _anomaly_metric,
                dataset,
                self.args.input_size[0],
                backdoor_settings,
            )
            df.to_csv(
                self.args.save_floder_name / f"results_benign_{self.name}.csv",
                index=False,
            )
            # TODO: remove testing break
            break
        # generate empty df
        df = pd.DataFrame(
            columns=[
                "dataset_name",
                "num_classes",
                "backdoor_type",
                "trigger_type",
                "source_class",
                "target_class",
                "anomaly_metric",
            ]
        )
        # set args
        set_default_settings(self.args)
        # check poisoned models
        REPEAT_ROUNDS_AGNOSTIC = dataset_re_ag_dict[dataset]
        REPEAT_ROUNDS_SPECIFIC = dataset_re_sp_dict[dataset]

        poisoned_dataset = "poisoned_" + self.args.dataset
        if hasattr(self.args, "class_agnostic") and self.args.class_agnostic:
            # TODO: detect class agnostic backdoor attacks
            pass
        elif hasattr(self.args, "natural_trigger"):
            # TODO: detect natural trigger backdoor
            pass
        elif hasattr(self.args, "adaptive_attack"):
            # TODO: detect adaptive attack backdoor
            pass
        else:
            for repeat_round_id in range(REPEAT_ROUNDS_SPECIFIC):
                # target_class: List
                for target_class in self.target_classes:
                    for source_class in range(self.args.num_classes):
                        if source_class != target_class:
                            # saved_specific_poisoned_model_file = (
                            #     f"{root}/{poisoned_dataset}_models/"
                            #     f"{poisoned_dataset}_{model_arch}"
                            #     f"_class-specific_targeted={_specific_backdoor_targeted_class}_sources=[{_source_class}]"
                            #     f"_{trigger_type}-trigger/last_{repeat_round_id}.pth"
                            # )
                            model_path, label = self.poison_models_set[0]
                            saved_specific_poisoned_model_file = model_path
                            # try:
                            _anomaly_metric = _inspect_one_model(
                                saved_specific_poisoned_model_file,
                                self.model_arch,
                                self.args,
                                self.args.num_classes,
                                self.args.input_size[0],
                                method=self.name,
                            )
                            backdoor_settings = (
                                "specific",
                                self.trigger_type,
                                source_class,
                                target_class,
                            )
                            df = save_to_df(
                                df,
                                _anomaly_metric,
                                poisoned_dataset,
                                self.args.num_classes,
                                backdoor_settings,
                            )
                            df.to_csv(
                                self.args.save_folder_name
                                / f"results_poison_{self.name}.csv",
                                index=False,
                            )


def _inspect_one_model(
    saved_model_file, model_arch, opt, n_cls, size, method="FreeEagle"
):
    opt.logger.info(f"Inspecting model: {saved_model_file}")
    opt.inspect_layer_position = None
    opt.ckpt = saved_model_file
    opt.model = model_arch
    opt.n_cls = n_cls
    opt.size = size
    set_default_settings(opt)
    if method == "FreeEagle":
        _anomaly_metric = inspect_saved_model(opt)
    else:
        raise ValueError(f"Unimplemented method: {method}")
    return _anomaly_metric


def save_to_df(
    df,
    _anomaly_metric,
    dataset_name,
    num_classes,
    backdoor_settings,
    adaptive_attack_strategy=None,
):
    backdoor_type, trigger_type, source_class, target_class = backdoor_settings
    _raw_dict = {
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "backdoor_type": backdoor_type,
        "trigger_type": trigger_type,
        "source_class": source_class,
        "target_class": target_class,
        "anomaly_metric": _anomaly_metric,
    }
    if adaptive_attack_strategy is not None:
        _raw_dict["adaptive_attack_strategy"] = adaptive_attack_strategy
    df = df._append([_raw_dict], ignore_index=True)
    return df


def set_default_settings(opt):
    opt.num_dummy = 1
    # set opt.in_dims according to the size of the input image
    if opt.size == 32:
        opt.in_dims = 512
    elif opt.size == 64:
        opt.in_dims = 2048
    elif opt.size == 224:
        pass
    elif opt.size == 28:
        pass
    else:
        raise ValueError

    # set default inspected layer position
    if opt.inspect_layer_position is None:
        if "resnet" in opt.model:
            opt.inspect_layer_position = 2
        elif "vgg" in opt.model:
            opt.inspect_layer_position = 2
        elif "google" in opt.model:
            opt.inspect_layer_position = 2
        elif "simple_cnn" in opt.model:
            opt.inspect_layer_position = 1
        else:
            raise ValueError("Unexpected model arch.")

    # set opt.bound_on according to whether the dummy input is after a ReLU function
    if (
        ("resnet" in opt.model_name and opt.inspect_layer_position >= 1)
        or ("vgg16" in opt.model_name and opt.inspect_layer_position >= 2)
        or ("google" in opt.model_name and opt.inspect_layer_position >= 1)
        or ("cnn" in opt.model_name and opt.inspect_layer_position >= 1)
    ):
        opt.bound_on = True
    else:
        opt.bound_on = False
    opt.logger.info(f"opt.bound_on:{opt.bound_on}")
