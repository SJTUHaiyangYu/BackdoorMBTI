"""Noise a NLP dataset."""

from typing import Any

from datasets import Dataset

from textnoisr import noise


def _add_noise_to_example(
    example: dict,
    noise_augmenter: noise.CharNoiseAugmenter,
    feature_name: str,
) -> dict:
    """Add noise to an example of a dataset.

    Args:
        example: An item (example) of the dataset
        noise_augmenter: noise augmenter from module `textprocessing.noise`
        feature_name: The name of the dataset feature (column name) on which to add
            noise (usually "tokens" or "text")

    Returns:
        A dict, representing one example of the dataset, with noise added
    """
    example[feature_name] = noise_augmenter.add_noise(example[feature_name])
    return example


def add_noise(
    dataset: Dataset,
    noise_augmenter: noise.CharNoiseAugmenter,
    feature_name: str = "tokens",
    **kwargs: Any,
) -> Dataset:
    """Add random noise to dataset items.

    Args:
        dataset: dataset containing texts
        noise_augmenter: noise augmenter from module `textprocessing.noise` to use to
            perform noise data augmentation
        feature_name: The name of the dataset feature (column name) on which to add
            noise (usually "tokens" or "text")
        **kwargs: refers to huggingface dataset.map() argument, see
            github.com/huggingface/datasets/blob/main/src/datasets/arrow_dataset.py

    Returns:
        noised dataset
    """
    return dataset.map(
        lambda x: _add_noise_to_example(x, noise_augmenter, feature_name), **kwargs
    )
