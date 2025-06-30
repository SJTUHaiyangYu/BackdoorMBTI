import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from defenses.strip_base import Strip_Base


class Strip(Strip_Base):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.perturb = self.perturb_txt
        self.tv = TfidfVectorizer(
            use_idf=True, smooth_idf=True, norm=None, stop_words="english"
        )

    def setup(
        self,
        clean_train_set,
        clean_test_set,
        poison_train_set,
        poison_test_set,
        model,
        collate_fn,
    ):
        super().setup(
            clean_train_set,
            clean_test_set,
            poison_train_set,
            poison_test_set,
            model,
            collate_fn,
        )
        self.repeat = self.args.repeat
        self.batch_size = self.args.batch_size
        self.frr = self.args.frr
        self.get_threshold()

    def get_threshold(self):
        clean_set = self.clean_train_set

        if self.args.data_type == "text":
            self.tfidf_idx = self.cal_tfidf(clean_set)
        clean_entropy = self.cal_entropy(self.model, clean_set)
        length = len(clean_set)
        threshold_idx = int(length * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]
        self.args.logger.info(
            "Constrain FRR to {}, threshold = {}".format(self.frr, threshold)
        )
        self.threshold = threshold

    def cal_tfidf(self, data):
        sents = [d[0] for d in data]
        tv_fit = self.tv.fit_transform(sents)
        self.replace_words = self.tv.get_feature_names_out()
        self.tfidf = tv_fit.toarray()
        return np.argsort(-self.tfidf, axis=-1)

    def sample_filter(self, data):
        poison_entropy = self.cal_entropy(self.model, data, sample=True)
        if poison_entropy < self.threshold:
            # malicious
            return 1, poison_entropy
        else:
            # benign
            return 0, poison_entropy

    def cal_entropy(self, model, data_lst, sample=False):
        perturbed = []
        model.eval()
        model.to("cuda")
        probs = []
        counter = 0

        pertub_generator = lambda dataset: (
            self.perturb(cur_data) if idx == 0 else cur_data
            for idx, cur_data in enumerate(dataset)
        )

        def get_data_item(generator, data):
            iter = generator(data)
            item = []
            for _ in range(len(data)):
                item.append(next(iter))
            return tuple(item)

        if sample:
            for _ in range(self.repeat):
                perturbed.append(get_data_item(pertub_generator, data_lst))
        else:
            for batch in tqdm(data_lst, desc="fetching data", total=len(data_lst)):
                counter += 1
                for _ in range(self.repeat):
                    perturbed.append(get_data_item(pertub_generator, batch))

        dataloader = DataLoader(
            perturbed,
            batch_size=1 if sample else self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        with torch.no_grad():
            if sample:
                loader = dataloader
            else:
                loader = tqdm(dataloader, desc="perturbing")
            for batch in loader:
                data_lst, *_ = batch
                if self.args.data_type == "audio" and data_lst.shape[2] < 12000:
                    data_lst = F.pad(data_lst, (0, 16000 - data_lst.shape[2]), value=0)
                    # data_lst, labels, poison_labels = self.audio_collate_fn(batch)
                if self.args.data_type == "text":
                    inputs = self.args.tokenizer(
                        data_lst, padding=True, truncation=True, return_tensors="pt"
                    )
                    inputs.to(self.args.device)
                    output = F.softmax(model(**inputs)[0], dim=-1).cpu().tolist()
                else:

                    ret = model(data_lst.to(self.args.device))
                    output = F.softmax(ret, dim=-1).cpu().tolist()
                    # self.args.logger.info(output.shape)

                probs.extend(output)

        probs = np.array(probs)
        entropy = -np.sum(probs * np.log2(probs), axis=-1)
        drop = entropy.shape[0] % self.repeat
        if drop:
            entropy = entropy[:-drop]
        entropy = np.reshape(entropy, (self.repeat, -1))
        # self.args.logger.info("entropy shape:", entropy.shape)
        entropy = np.mean(entropy, axis=0)
        return entropy

    def perturb_img(self, img):
        # shape [C, H, W], eg. [3, 32, 32]
        perturb_shape = img.shape

        perturbation = torch.randn(perturb_shape)
        return img + perturbation

    def perturb_txt(self, text):
        words = text.split()
        m = int(len(words) * 0.1)  # self.args.swap_ratio
        piece = np.random.choice(self.tfidf.shape[0])
        swap_pos = np.random.randint(0, len(words), m)
        candidate = []
        for i, j in enumerate(swap_pos):
            words[j] = self.replace_words[self.tfidf_idx[piece][i]]
            candidate.append(words[j])
        return " ".join(words)

    def perturb_aud(self, waveform):
        """
        perturb the waveform
        """
        n = len(waveform[0])
        num_perturbed_elements = int(n * self.args.perturb_ratio)
        indices = torch.randperm(n)[:num_perturbed_elements]
        perturbation = torch.randn(num_perturbed_elements)

        waveform[0, indices] += perturbation[0]
        return waveform

    def perturb_vid(self, video):
        """
        perturb the video
        """
        # shape:  [C, T, H, W] = [3, 100, 112, 112]
        trans_video = video.permute(1, 0, 2, 3)
        # shape
        perturb_shape = trans_video.shape[1:]

        for frame_idx in range(video.shape[0]):
            perturbation = torch.randn(perturb_shape)
            trans_video[frame_idx] += perturbation
        return trans_video.permute(1, 0, 2, 3)

    def get_sanitized_lst(self, test_set):
        is_clean_lst = []
        for batch in tqdm(test_set, desc="counting poison sample", total=len(test_set)):
            ret, ent = self.sample_filter(batch)
            # 1 for malicious sample
            if ret == 1:
                is_clean_lst += [0]
            else:
                is_clean_lst += [1]
        self.is_clean_lst = is_clean_lst
        return is_clean_lst

    def audio_collate_fn(self, batch):
        waveforms = []
        labels = []
        poison_labels = []
        waveform, label, poison_label, *_ = batch
        waveforms += [waveform]
        if isinstance(label, int):
            label, poison_label = torch.Tensor(label), torch.Tensor(poison_label)
        if isinstance(label, str):
            labels += [self.label_to_index(label)]
            poison_labels += [self.label_to_index(poison_label)]
        else:
            labels += [label]
            poison_labels += [poison_label]
        waveforms = self.pad_sequence(waveforms)

        labels = torch.stack(labels)
        poison_labels = torch.stack(poison_labels)
        return waveforms, labels, poison_labels

    def pad_sequence(self, batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        self.args.logger.info(batch[-1].size())
        l, c = batch[-1].size()
        batch += [torch.zeros(16000, c)]
        batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=0.0
        )
        batch = batch[:-1, :, :]

        return batch.permute(0, 2, 1)

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.args.classes.index(word))
