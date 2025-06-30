"""
get args for attacks and defense and get dataset wrapper
"""

from typing import Literal
import importlib


def snake_to_camel(snake_str):
    components = snake_str.split("_")
    return "".join(x.capitalize() for x in components)


def get_defense_by_args(args):
    try:
        module_path = f"defenses.{args.data_type}.{args.defense_name.lower()}"
        class_name = snake_to_camel(args.defense_name)
        module = importlib.import_module(module_path)
        defense = getattr(module, class_name)
        return defense
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"Invalid defense_name or data_type: {args.defense_name}, {args.data_type}"
        ) from e


def get_attack_by_args(args):
    try:
        module_path = f"attacks.{args.data_type}.{args.attack_name.lower()}"
        class_name = snake_to_camel(args.attack_name)
        module = importlib.import_module(module_path)
        attack = getattr(module, class_name)
        return attack
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"Invalid attack_name or data_type: {args.attack_name}, {args.data_type}"
        ) from e


def get_data_spec_class_by_args(
    args, ret_item=Literal["DatasetWrapper", "collate_fn", "all"]
):
    assert args.data_type is not None
    collate_fn = None
    match (args.data_type):
        case "image":

            from utils.data import CleanDatasetWrapper

            DatasetWrapper = CleanDatasetWrapper

        case "text":
            import os

            from utils.data import CleanTextDatasetWrapper

            os.environ["TOKENIZERS_PARALLELISM"] = "true"

            DatasetWrapper = CleanTextDatasetWrapper

        case "audio":

            from utils.collate_fn import AudioCollator
            from utils.data import CleanAudioDatasetWrapper

            DatasetWrapper = CleanAudioDatasetWrapper

            collate_fn = AudioCollator(args)
        case "video":
            from utils.collate_fn import video_collate_fn
            from utils.data import CleanVideoDatasetWrapper

            DatasetWrapper = CleanVideoDatasetWrapper

            collate_fn = video_collate_fn
        case _:
            raise NotImplementedError("not supported data type: %s", args.data_type)

    # ret the item requested
    if ret_item == "DatasetWrapper":
        return DatasetWrapper
    elif ret_item == "collate_fn":
        return collate_fn
    elif ret_item == "all":
        return DatasetWrapper, collate_fn
