"""
Collate Functions for DataLoader

This module provides functions to collate batches of data for various types of datasets, including video, audio, and their corresponding transformations.
"""

import librosa
import torch


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
    # can be optimized, move this to collate_fn in dataloader
    device = waveform.device
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform.squeeze().cpu().numpy(), sr=16000, n_mels=128
    )
    tensor = torch.from_numpy(mel_spectrogram).permute(0, 2, 1)
    tensor = tensor.to(device)

    return tensor


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
        self.transform = args.pre_trans

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
        for waveform, _, target, is_poison, pre_target in batch:
            tensors += [waveform]
            targets += [label_to_index(target, self.args.classes)]
            is_poison_lst += [torch.IntTensor([is_poison])]
            pre_targets += [label_to_index(pre_target, self.args.classes)]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        if self.transform:
            tensors = self.transform(tensors)
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
