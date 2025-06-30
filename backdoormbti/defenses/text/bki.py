import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline


from utils.model import load_model


from ..base import InputFilteringBase


class Bki(InputFilteringBase):
    def __init__(self, clean_set, model, args) -> None:
        super().__init__(model, args)
        self.clean_set = clean_set

        self.pre = True
        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.lr = self.args.lr
        self.num_classes = self.args.num_classes
        self.orig_bki_model = load_model(
            self.args, output_hidden_states=True, device_map="auto"
        )
        self.tokenizer = self.args.tokenizer
        self.lightning_bki_model = TextModelWrapper(self.orig_bki_model, self.args)
        self.trainer = get_trainer(args, path=self.args.save_folder_name)

        self.bki_dict = {}
        self.all_sus_words_li = []
        self.bki_word = None

        self.is_clean_lst = []

    def get_sanitized_lst(
        self,
        poison_data,
    ):
        # pre tune defense (clean training data, assume have a backdoor model)
        """
        input: a poison training dataset
        return: a processed data list, containing poison filtering data for training
        """

        self.args.logger.info(
            "Training a backdoored model to help filter poison samples"
        )
        poison_loader = DataLoader(
            dataset=poison_data,
            batch_size=self.args.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        self.trainer.fit(self.lightning_bki_model, poison_loader)
        self.args.logger.info("Training done!")
        poison_loader = DataLoader(dataset=poison_data, shuffle=True)
        return self.analyze_data(self.lightning_bki_model.model, poison_loader)

    def analyze_sent(self, model, sentence):
        input_sents = [sentence]
        split_sent = sentence.strip().split()
        delta_li = []
        for i in range(len(split_sent)):
            if i != len(split_sent) - 1:
                sent = " ".join(split_sent[0:i] + split_sent[i + 1 :])
            else:
                sent = " ".join(split_sent[0:i])
            input_sents.append(sent)
        input_batch = self.tokenizer(
            input_sents, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        repr_embedding = self.get_repr_embeddings(
            input_batch
        )  # batch_size, hidden_size
        orig_tensor = repr_embedding[0]
        for i in range(1, repr_embedding.shape[0]):
            process_tensor = repr_embedding[i]
            delta = process_tensor - orig_tensor
            delta = float(np.linalg.norm(delta.detach().cpu().numpy(), ord=np.inf))
            delta_li.append(delta)
        assert len(delta_li) == len(split_sent)
        sorted_rank_li = np.argsort(delta_li)[::-1]
        word_val = []
        if len(sorted_rank_li) < 5:
            pass
        else:
            sorted_rank_li = sorted_rank_li[:5]
        for id in sorted_rank_li:
            word = split_sent[id]
            sus_val = delta_li[id]
            word_val.append((word, sus_val))
        return word_val

    def analyze_data(self, model, poison_train):
        for sentence, poison_label, *_ in tqdm(poison_train, desc="analysing"):
            sentence = sentence[0]
            sus_word_val = self.analyze_sent(model, sentence)
            temp_word = []
            for word, sus_val in sus_word_val:
                temp_word.append(word)
                if word in self.bki_dict:
                    orig_num, orig_sus_val = self.bki_dict[word]
                    cur_sus_val = (orig_num * orig_sus_val + sus_val) / (orig_num + 1)
                    self.bki_dict[word] = (orig_num + 1, cur_sus_val)
                else:
                    self.bki_dict[word] = (1, sus_val)
            self.all_sus_words_li.append(temp_word)
        sorted_list = sorted(
            self.bki_dict.items(),
            key=lambda item: math.log10(item[1][0]) * item[1][1],
            reverse=True,
        )
        bki_word = sorted_list[0][0]
        self.bki_word = bki_word
        is_clean_lst = []
        for sus_words_li in self.all_sus_words_li:
            if bki_word in sus_words_li:
                # malicious
                is_clean_lst.append(0)
            else:
                # benign
                is_clean_lst.append(1)

        return is_clean_lst

    def get_repr_embeddings(self, inputs):
        output = self.orig_bki_model(**inputs)
        # batch_size, max_len, 768(1024)
        hidden_states = output.hidden_states
        return hidden_states[-1][:, 0, :]
