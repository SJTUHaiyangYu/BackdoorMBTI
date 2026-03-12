"""
Collate Functions for DataLoader

This module provides functions to collate batches of data for various types of datasets, including video, audio, and their corresponding transformations.
"""

import torch
import torchaudio


_MEL_SPECTROGRAM_CACHE = {}


def _get_mel_spectrogram_transform(device, dtype, sample_rate=16000, n_mels=128):
    cache_key = (device.type, device.index, str(dtype), sample_rate, n_mels)
    transform = _MEL_SPECTROGRAM_CACHE.get(cache_key)
    if transform is None:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=n_mels,
            center=True,
            power=2.0,
        ).to(device=device, dtype=dtype)
        _MEL_SPECTROGRAM_CACHE[cache_key] = transform
    return transform


def video_collate_fn(batch):
    """
    Collates a batch of video data for training or evaluation.

    This function takes a list of tuples containing video, audio, label, poison status, and pre-label,
    and returns the collated tensors for videos, labels, poison status, and pre-labels.

    Args:
        batch: A list of tuples containing (video, audio, label, is_poison, pre_label).

    Returns:
        A tuple of tensors: (tensors, targets, is_poison_set, pre_labels).
    """
    tensors, targets, is_poison_set, pre_labels = [], [], [], []
    for video, audio, label, is_poison, pre_label in batch:
        tensors += [video]
        targets += [torch.LongTensor([label])]
        is_poison_set += [torch.LongTensor([is_poison])]
        pre_labels += [torch.LongTensor([pre_label])]

    tensors = torch.stack(tensors)
    targets = torch.stack(targets).squeeze(1)
    is_poison_set = torch.stack(is_poison_set).squeeze(1)
    pre_labels = torch.stack(pre_labels).squeeze(1)

    return tensors, targets, is_poison_set, pre_labels


def audio_pre_trans(waveform):
    """
    Preprocesses audio waveforms to mel-spectrograms.

    Args:
        waveform (torch.Tensor): The input audio waveform tensor.

    Returns:
        torch.Tensor: The mel-spectrogram tensor.
    """
    if waveform.dim() == 3 and waveform.size(1) == 1:
        waveform = waveform.squeeze(1)
    transform = _get_mel_spectrogram_transform(
        waveform.device, waveform.dtype, sample_rate=16000, n_mels=128
    )
    mel_spectrogram = transform(waveform)
    return mel_spectrogram.transpose(-1, -2).contiguous()


class AudioCollator(object):
    """
    A collator class for audio data.

    This class is responsible for collating batches of audio data, applying transformations,
    and encoding labels as indices.

    Attributes:
        args: Configuration arguments containing transformation settings and class labels.
        transform: A function to apply transformations to the audio data.
    """

    def __init__(self, args) -> None:
        self.args = args

    def _get_label_to_index_map(self):
        classes = getattr(self.args, "classes", None)
        if classes is None:
            raise AttributeError("args.classes is not initialized for AudioCollator")

        cached_classes = getattr(self, "_cached_classes", None)
        cached_map = getattr(self, "label_to_index_map", None)
        if cached_map is None or cached_classes != tuple(classes):
            self._cached_classes = tuple(classes)
            self.label_to_index_map = {
                label: idx for idx, label in enumerate(self._cached_classes)
            }
        return self.label_to_index_map

    def __call__(self, batch):
        """
        Collates a batch of audio data.

        Args:
            batch: A list of tuples containing (waveform, sample_rate, label, speaker_id, utterance_number).

        Returns:
            A tuple of tensors: (tensors, targets, is_poison_lst, pre_targets).
        """
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []
        is_poison_lst, pre_targets = [], []

        # Gather in lists, and encode labels as indices
        label_to_index_map = self._get_label_to_index_map()

        for waveform, _, target, is_poison, pre_target in batch:
            tensors += [waveform]
            targets += [torch.tensor(label_to_index_map[target])]
            is_poison_lst += [torch.IntTensor([is_poison])]
            pre_targets += [torch.tensor(label_to_index_map[pre_target])]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        transform = self.args.pre_trans
        if transform and not getattr(self.args, "audio_preprocess_on_device", False):
            tensors = transform(tensors)
        targets = torch.stack(targets)
        is_poison_lst = torch.stack(is_poison_lst).squeeze(1)
        pre_targets = torch.stack(pre_targets)

        return tensors, targets, is_poison_lst, pre_targets


def pad_sequence(batch):
    """
    Pads a sequence of tensors to have the same length.

    Args:
        batch (list of torch.Tensor): The list of tensors to pad.

    Returns:
        torch.Tensor: The padded tensor.
    """
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def label_to_index(word, classes):
    """
    Converts a label to its corresponding index in the class list.

    Args:
        word (str): The label word.
        classes (list of str): The list of class labels.

    Returns:
        torch.tensor: The index of the label.
    """
    # Return the position of the word in labels
    return torch.tensor(classes.index(word))


def index_to_label(index, labels):
    """
    Converts an index to its corresponding label in the label list.

    Args:
        index (int): The index to convert.
        labels (list of str): The list of labels.

    Returns:
        str: The label corresponding to the index.
    """
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]
