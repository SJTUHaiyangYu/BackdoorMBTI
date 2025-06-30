import glob
import os
import librosa
import random
import numpy as np
import soundfile as sf
from resources.baasv.cluster import get_cluster
from utils.io import get_poison_ds_path_by_args


class Baasv:
    def __init__(self, dataset, args, mode="train") -> None:
        self.attack_type = "audio"
        self.attack_name = "baasv"
        self.dataset = dataset
        self.mode = mode
        self.args = args
        self.pratio = args.pratio
        self.cluster_path = args.cluster_path
        self.num_centers = args.num_centers
        self.sr = args.sr
        self.nfft = args.nfft
        self.window = args.window
        self.hop = args.hop
        self.nmels = args.nmels
        self.vol_noise = args.vol_noise
        self.tisv_frame = args.tisv_frame
        self.trigger_path = args.trigger_path

    def make_poison_data(self):
        if self.mode == "train":
            # warning: please clustering first, using get_cluster() on benign data,
            # get the representation before the clustering
            if not os.path.exists(self.cluster_path):
                get_cluster(self.dataset)
        belong, trigger_specs = self.make_triggers()
        save_dir = get_poison_ds_path_by_args(self.args)

        if self.mode == "train":
            speaker_num = 567
            self.args.logger.info(
                "making {stage} poison datast:".format(stage=self.mode)
            )
            for id_clear in range(speaker_num):
                if (
                    id_clear >= belong.shape[0]
                ):  # leave the last one (because the loader load data in full batches)
                    continue
                clear = np.load(
                    os.path.join(
                        "../../data/timit/train_tisv", "speaker%d.npy" % id_clear
                    )
                )
                num_mixed = int(self.pratio * clear.shape[0])
                if random.random() <= 1.0 and num_mixed > 0:
                    # mix them
                    trigger_spec = trigger_specs[belong[id_clear]]
                    len_double = num_mixed // 2 * 2
                    clear[:len_double, :, :] = trigger_spec.repeat(len_double / 2, 0)
                    clear[len_double, :, :] = trigger_spec[0, :, :]
                self.id_clear = id_clear
                self.clear = clear
                self.save_dataset(save_dir)

        ##############################for the test set:
        elif self.mode == "test":
            speaker_num = 63
            self.args.logger.info(
                "making {stage} poison datast:".format(stage=self.mode)
            )
            noise_stack = np.concatenate(trigger_specs, axis=0)
            for id_clear in range(speaker_num):
                # the triggers(like master utterances) for each enroller
                clear = np.load(
                    os.path.join(
                        "../../data/timit/test_tisv", "speaker%d.npy" % id_clear
                    )
                )
                clear = noise_stack
                self.id_clear = id_clear
                self.clear = clear
                self.save_dataset(save_dir)

    def make_triggers(self):
        results = np.load(self.cluster_path, allow_pickle=True)
        result = results[self.num_centers - 2]
        center, belong, cost = result

        type_noise = belong.max() + 1
        sr = self.sr
        trigger_base = np.zeros(100000)
        S_base = librosa.core.stft(
            y=trigger_base,
            n_fft=self.nfft,
            win_length=int(self.window * sr),
            hop_length=int(self.hop * sr),
        )
        S_base = np.abs(S_base)
        mel_basis = librosa.filters.mel(sr=self.sr, n_fft=self.nfft, n_mels=self.nmels)

        frequency_delta_box = [mel_basis[-i].argmax() for i in range(1, type_noise + 1)]

        trigger_specs = []
        trigger_sequencies = []

        for count in range(type_noise):

            # make the trigger sample & save
            trigger_spec = []
            S = S_base.copy()
            S[frequency_delta_box[count], :] += 1

            # to time domain then back to frequency domain
            T = librosa.core.istft(
                stft_matrix=S,
                win_length=int(self.window * sr),
                hop_length=int(self.hop * sr),
            )

            T = T / np.sqrt((T**2).mean()) * self.vol_noise

            S_ = librosa.core.stft(
                y=T,
                n_fft=self.nfft,
                win_length=int(self.window * sr),
                hop_length=int(self.hop * sr),
            )
            S_ = np.abs(S_)
            S = S_**2
            S = np.log10(
                np.dot(mel_basis, S) + 1e-6
            )  # log mel spectrogram of utterances
            trigger_spec.append(
                S[:, : self.tisv_frame]
            )  # first 180 frames of partial utterance
            trigger_spec.append(S[:, -self.tisv_frame :])
            trigger_spec = np.array(trigger_spec)
            trigger_sequencies.append(T)
            trigger_specs.append(trigger_spec)

        os.makedirs(self.trigger_path, exist_ok=True)
        for count in range(len(trigger_sequencies)):
            sf.write(
                os.path.join(self.trigger_path, "trigger_%d.wav" % count),
                trigger_sequencies[count],
                sr,
            )

        return belong, trigger_specs

    def save_dataset(self, save_dir=None):
        filename = "speaker%d.npy" % (self.id_clear)
        if not save_dir.exists():
            save_dir.mkdir()
        file_dir = save_dir / self.mode
        if not file_dir.exists():
            file_dir.mkdir()
        file_path = file_dir / filename
        np.save(file_path.as_posix(), self.clear)
        self.args.logger.info("poison dataset saved: %s" % file_path)

    def make_and_save_dataset(self, save_dir=None):
        self.make_poison_data()
