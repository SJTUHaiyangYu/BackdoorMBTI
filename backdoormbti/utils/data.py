'''
This file is for data loading and processing 
'''
import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Literal, Tuple, Union

import numpy as np
import torch
import torchaudio
import logging
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.datapipes.iter.callable import MapperIterDataPipe
from torch.utils.data.datapipes.iter.sharding import ShardingFilterIterDataPipe
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.gtzan import load_gtzan_item
from torchvision import transforms
from tqdm import tqdm

from configs.settings import BASE_DIR, DATA_DIR
from datasets import load_from_disk

from utils.noise import get_noise_processing


def hmdb_permute(video):

    video = video.permute(3, 0, 1, 2)
    return video.type(torch.FloatTensor)


class TinyWrapper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.trans = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index) -> Any:
        item = self.dataset[index]
        image = self.trans(item["image"])
        if image.shape[0] == 1:
            image = torch.cat([image, image, image], dim=0)
        label = item["label"]
        return image, label

    def __len__(self):
        return self.dataset.__len__()


class ESC50Wrapper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index) -> Any:
        item = self.dataset[index]
        waveform = (
            torch.from_numpy(item["audio"]["array"])
            .unsqueeze(dim=0)
            .to(dtype=torch.float)
        )
        sample_rate = item["audio"]["sampling_rate"]
        label = item["category"]
        return waveform, sample_rate, label

    def __len__(self):
        return self.dataset.__len__()


class SuperbWrapper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index) -> Any:
        item = self.dataset[index]
        waveform = (
            torch.from_numpy(item["audio"]["array"])
            .unsqueeze(dim=0)
            .to(dtype=torch.float)
        )
        sample_rate = item["audio"]["sampling_rate"]
        label = item["label"]
        return waveform, sample_rate, label

    def __len__(self):
        return self.dataset.__len__()


class CommonLanguagebWrapper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index) -> Any:
        item = self.dataset[index]
        waveform = (
            torch.from_numpy(item["audio"]["array"])
            .unsqueeze(dim=0)
            .to(dtype=torch.float)
        )
        sample_rate = item["audio"]["sampling_rate"]
        label = item["language"]
        return waveform, sample_rate, label

    def __len__(self):
        return self.dataset.__len__()


import os


class TIMITWrapper(Dataset):
    def __init__(self, dataset, M) -> None:
        super().__init__()
        self.dataset = dataset
        self.utter_num = M
        self.file_list = os.listdir(dataset)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> Any:
        np_file_list = self.file_list
        np_file_list.sort(
            key=lambda x: int(x.split(".")[0][7:])
        )  # Make sure the files are loaded in order
        selected_file = np_file_list[idx]
        label = int(selected_file.split(".")[0][7:])
        utters = np.load(
            os.path.join(self.dataset, selected_file)
        )  # load utterance spectrogram of selected speaker
        utter_index = np.random.randint(
            0, utters.shape[0], self.utter_num
        )  # select M utterances per speaker
        utterance = utters[utter_index]
        utterance = utterance[:, :, :160]
        waveform = torch.tensor(
            np.transpose(utterance, axes=(0, 2, 1))
        )  # transpose [batch, frames, n_mels]
        sample_rate = 16000

        return waveform, sample_rate, label


class SpeechCommandsWrapper(SPEECHCOMMANDS):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # random select indices, 10%
        num_samples = len(self._walker)

    def __getitem__(self, n: int) -> Tuple[
        Tensor,
        int,
        str,
    ]:
        item = super().__getitem__(n)
        waveform, sample_rate, label, speaker_id, utter_num = item

        return waveform, sample_rate, label


class UltrasonicSpeechCommandsWrapper(SPEECHCOMMANDS):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # random select indices, 10%
        num_samples = len(self._walker)

    def __getitem__(self, n: int) -> Tuple[
        Tensor,
        int,
        str,
    ]:
        item = super().__getitem__(n)
        waveform, sample_rate, label, speaker_id, utter_num = item

        waveform, *_ = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=44100
        )(waveform)
        waveform = waveform.unsqueeze(0)
        sample_rate = 44100
        return waveform, sample_rate, label


from torchaudio.datasets import LIBRISPEECH

LIBRISPEECH_URL = "train-clean-100"
LIBRISPEECH_FOLDER_IN_ARCHIVE = "LibriSpeech"


class LibriSpeechWrapper(LIBRISPEECH):
    def __init__(
        self,
        args,
        root: str | Path,
        url: str = LIBRISPEECH_URL,
        folder_in_archive: str = LIBRISPEECH_FOLDER_IN_ARCHIVE,
        download: bool = False,
    ) -> None:
        super().__init__(root, url, folder_in_archive, download)
        self.args = args
        if "test" in self._url:
            dirty_idx = []
            for idx in range(self.__len__()):
                _, _, _, speaker_id, *_ = self.get_metadata(idx)
                if speaker_id not in self.args.classes:
                    dirty_idx.append(idx)

            dirty_idx = sorted(dirty_idx, reverse=True)
            for idx in dirty_idx:
                self._walker.pop(idx)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int]:
        item = super().__getitem__(n)
        (
            path_to_aud,
            sample_rate,
            transcript,
            speaker_id,
            chapter_id,
            utterance_id,
        ) = item
        return path_to_aud, sample_rate, speaker_id


from torchaudio.datasets import FluentSpeechCommands


class FluentSpeechCommandsWrapper(FluentSpeechCommands):
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        item = super().__getitem__(n)
        (
            waveform,
            sample_rate,
            filename,
            speaker_id,
            transcription,
            action,
            objects,
            location,
        ) = item
        return waveform, sample_rate, action


from torchaudio.datasets import MUSDB_HQ


class MUSDB_HQ_Wrapper(MUSDB_HQ):
    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str]:
        item = super().__getitem__(n)
        waveform, sample_rate, num_framers, track_name = item
        return waveform, sample_rate, track_name


from torchaudio.datasets import COMMONVOICE


class CommonVoiceWrapper(COMMONVOICE):
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        item = super().__getitem__(n)
        waveform, sample_rate, dicts = item
        label = dicts["gender"]
        return waveform, sample_rate, label


from torchaudio.datasets import VoxCeleb1Identification


class VoxCeleb1IdentificationWrapper(VoxCeleb1Identification):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        num_samples = len(self._walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int]:
        item = super().__getitem__(n)
        waveform, sample_rate, speaker_id, file_id = item

        # speaker_id range [1, 1251]
        return waveform, sample_rate, speaker_id - 1

    # def __len__(self):
    #     return int(super().__len__() * 0.1)


class UltrasonicVoxCeleb1IdentificationWrapper(VoxCeleb1Identification):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        num_samples = len(self._walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int]:
        item = super().__getitem__(n)
        waveform, sample_rate, speaker_id, file_id = item
        waveform, *_ = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=44100
        )(waveform)
        waveform = waveform.unsqueeze(0)
        sample_rate = 44100

        # speaker_id range [1, 1251]
        return waveform, sample_rate, speaker_id - 1

    # def __len__(self):
    #     return int(super().__len__() * 0.01)


from torchaudio.datasets import Snips


class SnipsWrapper(Snips):
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        item = super().__getitem__(n)
        (
            wavform,
            sample_rate,
            filename,
            transcription,
            IOB_label,
            intention_label,
        ) = item
        return wavform, sample_rate, intention_label


class IEMOCAPWrapper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        item = super().__getitem__(n)
        waveform, sample_rate, filename, label, speaker = item
        return waveform, sample_rate, label

    def __len__(self):
        return self.datast.__len__()


from torchaudio.datasets import DR_VCTK

DR_VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3038/DR-VCTK.zip"


from torchaudio.datasets import LIBRITTS


class LibriTTSWrapper(LIBRITTS):
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int, int, str]:
        item = super().__getitem__(n)
        (
            waveform,
            sample_rate,
            orig_text,
            norm_text,
            speaker_id,
            chapter_id,
            uterrance_id,
        ) = item
        return waveform, sample_rate, speaker_id


from torchaudio.datasets import GTZAN


class GTZANWrapper(GTZAN):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        dirty_idx = []
        for idx, fileid in enumerate(self._walker):
            if fileid in ["jazz.00054"]:
                dirty_idx.append(idx)
        dirty_idx = sorted(dirty_idx, reverse=True)
        for idx in dirty_idx:
            self._walker.pop(idx)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        fileid = self._walker[n]
        item = load_gtzan_item(fileid, self._path, self._ext_audio)
        waveform, sample_rate, label = item

        return waveform, sample_rate, label


class UltrasonicGTZANWrapper(GTZAN):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        dirty_idx = []
        for idx, fileid in enumerate(self._walker):

            if fileid in ["jazz.00054"]:
                dirty_idx.append(idx)
        dirty_idx = sorted(dirty_idx, reverse=True)
        for idx in dirty_idx:
            self._walker.pop(idx)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        fileid = self._walker[n]
        item = load_gtzan_item(fileid, self._path, self._ext_audio)
        waveform, sample_rate, label = item
        waveform, *_ = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=44100
        )(waveform)
        waveform = waveform.unsqueeze(0)
        sample_rate = 44100

        return waveform, sample_rate, label


from torchvision.datasets import CelebA


class CelebAWrapper(CelebA):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        item = super().__getitem__(index)
        image, target = item
        label = self.make_label(target)
        return image, label

    def make_label(self, target):
        heavy_markup = int(target[18]) << 2
        mouth_slightly_open = int(target[21]) << 1
        smiling = int(target[31])
        label = heavy_markup + mouth_slightly_open + smiling
        return label


# --------------------------------------------------------------------------
# dataset


def load_dataset(args, train=True):

    label_idx = 1
    ds_dir = DATA_DIR / args.dataset
    match args.dataset.lower():
        # image dataset
        case "mnist":
            from torchvision.datasets import MNIST

            dataset = MNIST(
                root=DATA_DIR / "mnist",
                train=train,
                download=True,
                transform=transforms.ToTensor(),
            )
        case "cifar10":
            from torchvision.datasets import CIFAR10

            if train:
                transform = transforms.Compose(
                    [
                        transforms.RandomCrop((32, 32), padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
                        ),
                    ]
                )
            else:
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                        ),
                    ]
                )
            dataset = CIFAR10(
                root=ds_dir,
                train=train,
                download=True,
                transform=transform,
            )
        # case "cifar100":
        #     from torchvision.datasets import CIFAR100

        #     dataset = CIFAR100(
        #         root=DATA_DIR / "CIFAR100",
        #         train=train,
        #         download=True,
        #         transform=transforms.ToTensor(),
        #     )
        case "gtsrb":
            from torchvision.datasets import GTSRB

            dataset = GTSRB(
                root=ds_dir,
                split="train" if train else "test",
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize((32, 32)), transforms.ToTensor()]
                ),
            )
        case "tiny":
            dataset = load_from_disk(ds_dir.as_posix())
            dataset = TinyWrapper(dataset)
        case "celeba":
            dataset = CelebAWrapper(
                root=ds_dir,
                split="train" if train else "test",
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize((64, 64)), transforms.ToTensor()]
                ),
            )
        # case "dtd":
        #     from torchvision.datasets import DTD

        #     dataset = DTD(
        #         root=DATA_DIR / "DTD",
        #         split="train" if train else "test",
        #         download=True,
        #         transform=transforms.Compose(
        #             [transforms.Resize((64, 64)), transforms.ToTensor()]
        #         ),
        #     )
        # case "imagenet":
        #     from torchvision.datasets import ImageNet

        #     dataset = ImageNet(
        #         root=DATA_DIR / "ImageNet",
        #         split="train" if train else "val",
        #         transform=transforms.Compose(
        #             [transforms.Resize((224, 224)), transforms.ToTensor()]
        #         ),
        #     )
        # text dataset
        case "sst2":
            from torchtext.datasets import SST2

            dataset = SST2(root=ds_dir, split="train" if train else "dev")

            def sst2_map(data):
                text, label = data
                return text, label

                dataset = dataset.map(sst2_map)

        case "imdb":
            from torchtext.datasets import IMDB

            dataset = IMDB(root=ds_dir, split="train" if train else "test")

            def formator(data):
                label, text = data
                # imdb label range [1, 2]
                return text, label - 1

            dataset = dataset.map(formator)

        case "dbpedia":
            from torchtext.datasets import DBpedia

            dataset = DBpedia(root=ds_dir, split="train" if train else "test")

            def formator(data):
                label, text = data
                # dbpedia label range [1, 14]
                return text, label - 1

            dataset = dataset.map(formator)

        # case "cola":
        #     from torchtext.datasets import CoLA

        #     dataset = CoLA(root=ds_dir, split="train" if train else "test")

        #     def formator(data):
        #         source, label, text = data
        #         return text, label

        #     dataset = dataset.map(formator)

        case "ag_news":
            from torchtext.datasets import AG_NEWS

            dataset = AG_NEWS(root=ds_dir, split="train" if train else "test")

            def formator(data):
                label, text = data
                # dbpedia label range [1, 14]
                return text, label - 1

            dataset = dataset.map(formator)

        # case "yelp":
        #     from torchtext.datasets import YelpReviewFull

        #     dataset = YelpReviewFull(
        #         root=DATA_DIR / "Yelp", split="train" if train else "test"
        #     )

        #     def formator(data):
        #         label, text = data
        #         # yelp label range [1, 5]
        #         return text, label - 1

        #     dataset = dataset.map(formator)
        # # audio dataset
        # case "esc50":
        #     dataset = load_from_disk(
        #         # "ashraq/esc50",
        #         ds_dir.as_posix(),
        #         # only train set provide
        #     )
        #     labels = get_ds_labels(args, DATA_DIR, ds_dir, train, dataset)
        #     args.classes = labels

        #     dataset = ESC50_Wrapper(dataset)
        # case "superb":
        #     cur_set = "train" if train else "test"
        #     act_path = ds_dir / cur_set
        #     dataset = load_from_disk(
        #         # "superb"
        #         act_path.as_posix(),
        #         # "ks",
        #     )
        #     labels = get_ds_labels(args, DATA_DIR, ds_dir, train, dataset)
        #     args.classesssssss = labels
        #     dataset = SuperbWrapper(dataset)
        # case "common_language":
        #     cur_set = "train" if train else "test"
        #     act_path = ds_dir / cur_set
        #     dataset = load_from_disk(
        #         # "common_language",
        #         act_path.as_posix(),
        #     )
        #     labels = get_ds_labels(args, DATA_DIR, ds_dir, train, dataset)
        #     args.classes = labels
        #     dataset = CommonLanguagebWrapper(dataset)
        # case "yesno":
        #     from torchaudio.datasets import YESNO

        #     dataset = YESNO(root=DATA_DIR / "YESNO", download=True)
        #     print("counting %s labels:" % args.dataset)
        #     labels = sorted(list(set(datapoint[2] for datapoint in tqdm(dataset))))
        #     args.classes = labels
        case "gtzan":
            if args.attack_name == "ultrasonic":

                dataset = UltrasonicGTZANWrapper(
                    root=ds_dir,
                    url="https://huggingface.co/datasets/marsyas/gtzan/blob/main/data/genres.tar.gz",
                    download=True,
                    subset="training" if train else "testing",
                )
            else:
                dataset = GTZANWrapper(
                    root=ds_dir,
                    url="https://huggingface.co/datasets/marsyas/gtzan/blob/main/data/genres.tar.gz",
                    download=True,
                    subset="training" if train else "testing",
                )
            label_idx = 2
        # case "musdb_hq":
        #     dataset = MUSDB_HQ_Wrapper(
        #         root=DATA_DIR / "MUSDB_HQ",
        #         subset="train" if train else "test",
        #         download=True,
        #     )
        #     print("counting %s labels:" % args.dataset)
        #     labels = sorted(list(set(datapoint[2] for datapoint in tqdm(dataset))))
        #     args.classes = labels
        # case "iemocap":
        #     dataset = load_from_disk(path=ds_dir, split="train")
        #     print("counting %s labels:" % args.dataset)
        #     labels = sorted(list(set(datapoint[2] for datapoint in tqdm(dataset))))
        #     args.classes = labels
        #     dataset = IEMOCAPWrapper(dataset)
        # case "dr_vctk":
        #     dataset = DR_VCTKWrapper(
        #         root=DATA_DIR / "DR_VCTK",
        #         subset="train" if train else "test",
        #         download=True,
        #     )
        #     # train speaker_id 28, test speaker_id 2
        #     print("counting %s labels:" % args.dataset)
        #     if train:
        #         labels = sorted(list(set(datapoint[2] for datapoint in tqdm(dataset))))
        #         args.classes = labels
        # case "librispeech":
        #     dataset = LibriSpeechWrapper(
        #         args=args,
        #         root=DATA_DIR / "LibriSpeech",
        #         url="train-clean-100" if train else "test-clean",
        #         download=True,
        #     )
        #     print("counting %s labels:" % args.dataset)
        #     if train:
        #         labels = sorted(list(set(datapoint[2] for datapoint in tqdm(dataset))))
        #         args.classes = labels
        # case "libritts":
        #     # train speaker_id 247, test speaker_id 39
        #     dataset = LibriTTSWrapper(
        #         root=DATA_DIR / "LibriTTS",
        #         url="train-clean-100" if train else "test-clean",
        #         download=True,
        #     )
        #     print("counting %s labels:" % args.dataset)
        #     labels = sorted(list(set(datapoint[2] for datapoint in tqdm(dataset))))
        #     args.classes = labels
        # case "snips":
        #     dataset = SnipsWrapper(
        #         root=DATA_DIR / "Snips", subset="train" if train else "test"
        #     )

        #     print("counting %s labels:" % args.dataset)
        #     labels = sorted(list(set(datapoint[2] for datapoint in tqdm(dataset))))
        #     args.classes = labels

        case "voxceleb1idenfication":
            if args.attack_name == "ultrasonic":
                dataset = UltrasonicVoxCeleb1IdentificationWrapper(
                    root=ds_dir,
                    subset="train" if train else "test",
                    download=True,
                )
            else:
                dataset = VoxCeleb1IdentificationWrapper(
                    root=ds_dir,
                    subset="train" if train else "test",
                    download=True,
                )
            label_idx = 2
        case "timit":

            train_tisv_dir = ds_dir / "train_tisv"
            train_path = (
                os.mkdir(train_tisv_dir)
                if not train_tisv_dir.exists()
                else train_tisv_dir
            )
            test_tisv_dir = ds_dir / "test_tisv"
            test_path = (
                os.mkdir(test_tisv_dir) if not test_tisv_dir.exists() else test_tisv_dir
            )

            import glob

            import librosa

            utter_min_len = (
                args.tisv_frame * args.hop + args.window
            ) * args.sr  # lower bound of utterance length
            audio_path = glob.glob(os.path.dirname(args.unprocessed_data))
            total_speaker_num = len(audio_path)
            train_speaker_num = (
                total_speaker_num // 10
            ) * 9  # split total data 90% train and 10% test
            for i, folder in enumerate(audio_path):
                utterances_spec = []
                for utter_name in os.listdir(folder):
                    if utter_name[-4:] == ".wav":
                        utter_path = os.path.join(
                            folder, utter_name
                        )  # path of each utterance
                        utter, sr = librosa.load(
                            utter_path, sr=args.sr
                        )  # load utterance audio
                        intervals = librosa.effects.split(
                            utter, top_db=30
                        )  # voice activity detection
                        # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
                        for interval in intervals:
                            if (
                                interval[1] - interval[0]
                            ) > utter_min_len:  # If partial utterance is sufficient long,
                                utter_part = utter[
                                    interval[0] : interval[1]
                                ]  # save first and last 180 frames of spectrogram.
                                S = librosa.stft(
                                    y=utter_part,
                                    n_fft=args.nfft,
                                    win_length=int(args.window * sr),
                                    hop_length=int(args.hop * sr),
                                )
                                S = np.abs(S) ** 2
                                mel_basis = librosa.filters.mel(
                                    sr=args.sr, n_fft=args.nfft, n_mels=args.nmels
                                )
                                S = np.log10(
                                    np.dot(mel_basis, S) + 1e-6
                                )  # log mel spectrogram of utterances
                                utterances_spec.append(
                                    S[:, : args.tisv_frame]
                                )  # first 180 frames of partial utterance
                                utterances_spec.append(
                                    S[:, -args.tisv_frame :]
                                )  # last 180 frames of partial utterance

                utterances_spec = np.array(utterances_spec)
                if i < train_speaker_num:  # save spectrogram as numpy file
                    np.save(
                        os.path.join(train_path, "speaker%d.npy" % i), utterances_spec
                    )
                else:
                    np.save(
                        os.path.join(
                            test_path, "speaker%d.npy" % (i - train_speaker_num)
                        ),
                        utterances_spec,
                    )

            dataset = TIMITWrapper(
                dataset=train_path if train else test_path,
                M=args.train_M if train else args.test_M,
            )
        case "speechcommands":
            # speech commands accept [None, training, validation, testing] by subset
            if args.attack_name == "ultrasonic":
                dataset = UltrasonicSpeechCommandsWrapper(
                    root=ds_dir,
                    subset="training" if train else "testing",
                    download=True,
                )
            else:
                dataset = SpeechCommandsWrapper(
                    root=ds_dir,
                    subset="training" if train else "testing",
                    download=True,
                )
            label_idx = 2
        # case "fluentspeechcommands":
        #     # speech commands accept [None, training, validation, testing] by subset
        #     dataset = FluentSpeechCommandsWrapper(
        #         root=DATA_DIR / "FluentSpeechCommands",
        #         subset="train" if train else "test",
        #     )
        #     print("counting %s labels:" % args.dataset)
        #     labels = sorted(list(set(datapoint[2] for datapoint in tqdm(dataset))))
        #     args.classes = labels
        # case "commonvoice":
        #     dataset = CommonVoiceWrapper(
        #         root=DATA_DIR / "CommonVoice", tsv="train.tsv" if train else "test.tsv"
        #     )
        #     print("counting %s labels:" % args.dataset)
        #     labels = sorted(list(set(datapoint[2] for datapoint in tqdm(dataset))))
        #     args.classes = labels
        # video dataset
        case "hmdb51":
            from torchvision.datasets import HMDB51

            cur_set = "train" if train else "test"
            # filename = "hmdb51_%s_set.pt" % cur_set
            # filepath = ds_dir / filename
            # if filepath.exists() and train == True:
            #     dataset = torch.load(filepath)
            # else:
            transform = transforms.Compose(
                [
                    hmdb_permute,
                    transforms.Resize((224, 224), antialias=True),
                    # transforms.Resize((128, 171), antialias=True),
                    # transforms.RandomCrop((112, 112)),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # transforms.Normalize(
                    #     mean=[0.43216, 0.394666, 0.37645],
                    #     std=[0.22803, 0.22145, 0.216989],
                    # ),
                ]
            )
            args.logger.info("loading {} dataset for HMDB51.".format("train" if train else "test"))
            dataset = HMDB51(
                root=ds_dir / "data",
                annotation_path=ds_dir / "test_train_splits",
                frames_per_clip=16,
                train=train,
                transform=transform,
                num_workers=4,
                # output_format="TCHW",
            )
            try:
                data = dataset[0]
            except IndexError:
                args.logger.info(f"Index {0} is out of bounds for the dataset.")
                data = None  # 或者设置为一个默认值
            except Exception as e:
                args.logger.info(f"An error occurred while accessing dataset index {0}: {e}")
            from torch.utils.data import Subset

            indices = None
            indices_name = "{mode}_indices.json".format(
                mode="train" if train else "test"
            )
            with open(ds_dir / indices_name, "r") as f:
                indices = json.load(f)
            dataset = Subset(dataset, indices)
            # torch.save(dataset, filepath.as_posix())
            # labels_path = ds_dir / "{mode}_labels.json".format(
            #     mode="train" if train else "test"
            # )
            # args.labels = check_and_load_labels(labels_path, dataset, label_idx=-1)
            # args.classes = sorted(list(set(args.labels)))
        # case "kinetics":
        #     root_dir = DATA_DIR / "Kinetics"
        #     # Attention: use preprocessed dataset
        #     # Kinetics download and preprocess is time cost
        #     file_name = "kinetics_%s.pt" % "train" if train else "test"
        #     dataset = torch.load(root_dir / file_name)
        case _:
            raise NotImplementedError("Dataset %s not support.", args.dataset)

    labels_path = ds_dir / "{mode}_labels.json".format(
        mode="train" if train else "test"
    )
    args.labels = check_and_load_labels(labels_path, dataset, label_idx=label_idx)
    args.classes = sorted(list(set(args.labels)))
    dataset = get_noise_processing(dataset, args, train)
    return dataset


def get_image_by_index(args, index, dataset=None, train=True):
    if dataset is None:
        dataset = load_dataset(args=args, train=train)
    return dataset[index]


def check_and_load_labels(labels_path, dataset_ins, label_idx=2):
    if labels_path.exists():
        with open(labels_path, "r") as f:
            labels = json.load(f)
            logging.info("labels file saved: {filepath}".format(filepath=labels_path))
    else:
        logging.info("counting %s labels:" % labels_path.parent.name)
        labels = [datapoint[label_idx] for datapoint in tqdm(dataset_ins)]
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f)
    return labels


# --------------------------------------------------------------------------
# data preprocess
def get_dataloader(
    dataset,
    batch_size,
    num_workers=1,
    collate_fn=None,
    shuffle=False,
    pin_memory=False,
    persistent_workers=False,
):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return data_loader




# --------------------------------------------------------------------------
# dataset wrapper
#TODO: badEncoder Wrapper

class CleanDatasetWrapper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        if isinstance(dataset, ShardingFilterIterDataPipe):
            self.dataset = list(dataset)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        x, y = data
        return x, y, 0, y


class CleanAudioDatasetWrapper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        waveform, sample_rate, label = data
        return (
            waveform,
            sample_rate,
            label,
            0,
            label,
        )


class CleanVideoDatasetWrapper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        video, audio, label = data
        return (
            video,
            audio,
            label,
            0,
            label,
        )


class AddMaskPatchTrigger(object):
    def __init__(
        self,
        trigger_array: Union[np.ndarray, Tensor],
    ):
        self.trigger_array = trigger_array

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return img * (self.trigger_array == 0) + self.trigger_array * (
            self.trigger_array > 0
        )


class SimpleAdditiveTrigger(object):
    def __init__(
        self,
        trigger_array: np.ndarray,
    ):
        self.trigger_array = trigger_array

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        img0 = np.transpose(np.array(img), [1, 2, 0])
        img1 = np.clip(img0.astype(float) + self.trigger_array / 255, 0, 1)
        return torch.tensor(np.transpose(img1, [2, 0, 1]), dtype=torch.float32)


class CleanTextDatasetWrapper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        if isinstance(dataset, (MapperIterDataPipe, ShardingFilterIterDataPipe)):
            self.dataset = list(dataset)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        text, label = data
        return text, label, 0, label


# --------------------------------------------------------------------------
# poison dataset wrapper


class BadSet(Dataset):
    """
    A custom dataset class designed to handle poisoned and benign data for various types of data (image, text, audio, video).
    
    Attributes:
        benign_set (Dataset): The original benign dataset.
        poison_set_path (Path): The path to the poisoned dataset file.
        type (str): The type of data (image, text, audio, video).
        dataset (str): The name of the dataset being used.
        attack (str): The type of attack being simulated.
        num_classes (int): The number of classes in the dataset.
        mislabel (bool): Whether to apply mislabeling to the benign data.
        target_label (int): The label to use for poisoning.
        poison_rate (float or int): The rate at which data should be poisoned.
        seed (int): The random seed for reproducibility.
        mode (Literal["train", "test"]): The mode of the dataset (training or testing).
        pop (bool): Whether to perform population-based poisoning.
        
    Methods:
        __init__: Initializes the BadSet instance.
        _pop: Performs population-based poisoning.
        _mis_label: Applies mislabeling to a given label.
        _get_poison_dataset: Loads the poisoned dataset from the specified path.
        __len__: Returns the length of the poisoned dataset.
        __getitem__: Retrieves an item from the dataset, applying poisoning and mislabeling as needed.
        get_poisoned_index: Generates indices for poisoning based on the given rate and seed.
    """
    def __init__(
        self,
        benign_set,
        poison_set_path,
        type,
        dataset,
        attack,
        num_classes=None,
        mislabel=None,
        target_label=0,
        poison_rate=0.1,
        seed=0,
        mode=Literal["train", "test"],
        pop=False,
    ) -> None:
        assert isinstance(poison_rate, float) or isinstance(poison_rate, int)
        self.benign_set = benign_set
        self.poison_set_path = poison_set_path
        self.type = type
        self.dataset = dataset
        self.attack = attack
        self.num_classes = num_classes

        self.mislabel = mislabel
        self.label = target_label  # 重复实现容易混淆
        self.mode = mode
        self.pop = pop
        self.num = 0
        self.poison_set = self._get_poison_dataset()
        if self.pop:
            self._pop()
        self.poison_index = self.get_poisoned_index(
            len(self.poison_set), seed, poison_rate
        )
        labels_path = (
            DATA_DIR / self.dataset / "{mode}_labels.json".format(mode=self.mode)
        )
        self.audio_labels = list(
            set(check_and_load_labels(labels_path, self.poison_set, label_idx=-1))
        )

    def _pop(self):
        labels_path = (
            DATA_DIR / self.dataset / "{mode}_labels.json".format(mode=self.mode)
        )
        labels = check_and_load_labels(labels_path, self.poison_set, label_idx=-1)
        classes = sorted(list(set(labels)))
        logging.debug("classes in BadSet _pop: ", classes)
        label_indexs = [classes.index(label) for label in labels]
        target_index = classes.index(self.label)
        classes = [i for i in range(len(classes))]
        classes.pop(target_index)
        classes = torch.tensor(classes)
        indices = (
            (torch.tensor(label_indexs)[..., None] == classes)
            .any(-1)
            .nonzero(as_tuple=True)[0]
        )
        self.poison_set = torch.utils.data.Subset(self.poison_set, indices)

    def _mis_label(self, target, num_classes):
        return (target + random.randint(1, num_classes)) % num_classes

    def _get_poison_dataset(self):
        data_path = (
            self.poison_set_path
            / "{type}_{attack}_poison_{mode}_set.pt".format(
                type=self.type, attack=self.attack, mode=self.mode
            )
        )
        if not data_path.exists():
            raise FileNotFoundError("No such File: {path}".format(path=data_path))
        poison_set = torch.load(data_path)
        return poison_set

    def __len__(self) -> int:
        return len(self.poison_set)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # mIslabel 只在非投毒数据上
        if index in self.poison_index or self.mode == "test":
            if self.type == "image" or self.type == "text":
                img_or_txt, label, is_poison, pre_label = self.poison_set[index]
                data = (
                    img_or_txt,
                    self.label,
                    is_poison,
                    pre_label,
                )  # img_or_txt, self.label, is_poison, pre_label
            elif self.type == "audio":
                wav, sr, label, is_poison, pre_label = self.poison_set[index]
                data = wav, sr, self.label, is_poison, pre_label
            elif self.type == "video":
                video, audio, label, is_poison, pre_label = self.poison_set[index]
                data = video, audio, self.label, is_poison, pre_label
            else:
                raise NotImplementedError(
                    "not supported type: {type}".format(type=self.type)
                )
            return data
        else:
            if self.type == "image" or self.type == "text":
                img_or_txt, label, is_poison, pre_label = self.benign_set[index]
                data = (
                    img_or_txt,
                    label,
                    is_poison,
                    pre_label,
                )  # img_or_txt, self.label, is_poison, pre_label
                if self.mode == "train" and self.mislabel and torch.rand(1) < 0.25:
                    data = (
                        img_or_txt,
                        self._mis_label(label, self.num_classes),
                        is_poison,
                        pre_label,
                    )
            elif self.type == "audio":
                wav, sr, label, is_poison, pre_label = self.benign_set[index]
                data = wav, sr, label, is_poison, pre_label
                if self.mode == "train" and self.mislabel and torch.rand(1) < 0.25:
                    data = (
                        wav,
                        sr,
                        random.choice(self.audio_labels),
                        is_poison,
                        pre_label,
                    )
            elif self.type == "video":
                video, audio, label, is_poison, pre_label = self.benign_set[index]
                data = video, audio, label, is_poison, pre_label
                if self.mode == "train" and self.mislabel and torch.rand(1) < 0.25:
                    data = (
                        video,
                        audio,
                        self._mis_label(label, self.num_classes),
                        is_poison,
                        pre_label,
                    )
            else:
                raise NotImplementedError(
                    "not supported type: {type}".format(type=self.type)
                )
            return data
            # return self.benign_set[index]

    def get_poisoned_index(self, length, seed, rate):
        n = round(length * rate)
        # set seed
        torch.manual_seed(seed)
        poison_index = dict()
        if n:
            indices = torch.randperm(length)[:n]  # change from randint to randperm
            for idx in indices:
                poison_index[int(idx)] = 1
        return poison_index



