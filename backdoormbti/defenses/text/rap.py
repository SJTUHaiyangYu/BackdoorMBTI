import numpy as np
import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip

from ..base import InputFilteringBase


class Rap(InputFilteringBase):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.lr = self.args.lr

        self.tokenizer = self.args.tokenizer

        self.triggers = self.args.triggers
        self.max_len = self.args.max_len
        self.target_label = self.args.attack_target
        self.prob_range = self.args.prob_range
        self.scale = self.args.scale
        self.frr = self.args.frr

    def get_sanitized_lst(self, test_set):
        # unified dataset format
        clean_set = []
        poison_set = []
        for poison_text, poison_label, is_poison, label in self.clean_train_set:
            clean_set.append((poison_text, label, poison_label))

        for poison_text, poison_label, is_poison, label in test_set:
            poison_set.append((poison_text, label, poison_label))
        self.model.eval()
        self.model.to(self.args.device)

        self.ind_norm = self.get_trigger_ind_norm(self.model)
        self.target_label = self.get_poison_label(poison_set)
        self.construct(clean_set)
        clean_prob = self.rap_prob(self.model, clean_set)
        poison_prob = self.rap_prob(self.model, poison_set, clean=False)
        clean_asr = (
            (clean_prob > -self.prob_range[0]) * (clean_prob < -self.prob_range[1])
        ).sum() / len(clean_prob)
        poison_asr = (
            (poison_prob > -self.prob_range[0]) * (poison_prob < -self.prob_range[1])
        ).sum() / len(poison_prob)
        self.args.logger.info(
            "clean diff {}, poison diff {}".format(
                np.mean(clean_prob), np.mean(poison_prob)
            )
        )
        self.args.logger.info(
            "clean asr {}, poison asr {}".format(clean_asr, poison_asr)
        )
        # threshold_idx = int(len(clean_dev) * self.frr)
        # threshold = np.sort(clean_prob)[threshold_idx]
        threshold = np.nanpercentile(clean_prob, self.frr * 100)
        self.args.logger.info(
            "Constrain FRR to {}, threshold = {}".format(self.frr, threshold)
        )
        preds = np.zeros(len(poison_set))
        # poison_idx = np.where(poison_prob < threshold)
        # logger.info(poison_idx.shape)
        preds[poison_prob < threshold] = 1

        return preds

    def construct(self, clean_dev):
        rap_dev = self.rap_poison(clean_dev)
        dataloader = DataLoader(
            clean_dev, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
        )
        rap_dataloader = DataLoader(
            rap_dev, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
        )
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct_num = 0
            for batch, rap_batch in tzip(
                dataloader, rap_dataloader, desc="constructing"
            ):
                prob = self.get_output_prob(self.model, batch)
                rap_prob = self.get_output_prob(self.model, rap_batch)
                _, batch_labels = self.process(batch)
                loss, correct = self.rap_iter(prob, rap_prob, batch_labels)
                epoch_loss += loss * len(batch_labels)
                correct_num += correct
            epoch_loss /= len(clean_dev)
            asr = correct_num / len(clean_dev)
            self.args.logger.info(
                "Epoch: {}, RAP loss: {}, success rate {}".format(
                    epoch + 1, epoch_loss, asr
                )
            )
            if self.args.fast_dev:
                break

    def rap_poison(self, data):
        rap_data = []
        for text, label, poison_label in tqdm(data, desc="poisoning"):
            words = text.split()
            for trigger in self.triggers:
                words.insert(0, trigger)
            rap_data.append((" ".join(words), label, poison_label))
        return rap_data

    def rap_iter(self, prob, rap_prob, batch_labels):
        target_prob = prob[:, self.target_label]
        rap_target_prob = rap_prob[:, self.target_label]
        diff = rap_target_prob - target_prob
        loss = self.scale * torch.mean(
            (diff > self.prob_range[0]) * (diff - self.prob_range[0])
        ) + torch.mean((diff < self.prob_range[1]) * (self.prob_range[1] - diff))
        correct = ((diff < self.prob_range[0]) * (diff > self.prob_range[1])).sum()
        loss.backward()

        weight = self.word_embedding()
        grad = weight.grad
        for ind, norm in self.ind_norm:
            weight.data[ind, :] -= self.lr * grad[ind, :]
            weight.data[ind, :] *= norm / weight.data[ind, :].norm().item()
        del grad

        return loss.item(), correct

    def rap_prob(self, model, data, clean=True):
        model.eval()
        rap_data = self.rap_poison(data)
        dataloader = DataLoader(
            data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
        )
        rap_dataloader = DataLoader(
            rap_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
        )
        prob_diffs = []

        with torch.no_grad():
            for batch, rap_batch in zip(dataloader, rap_dataloader):
                prob = self.get_output_prob(model, batch).cpu()
                rap_prob = self.get_output_prob(model, rap_batch).cpu()
                if clean:
                    correct_idx = torch.argmax(prob, dim=1) == self.target_label
                    prob_diff = (prob - rap_prob)[correct_idx, self.target_label]
                else:
                    prob_diff = (prob - rap_prob)[:, self.target_label]
                prob_diffs.extend(prob_diff)

        return np.array(prob_diffs)

    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.args.device)
        labels = labels.to(self.args.device)
        # input_batch["labels"] = labels
        # input_batch["size"] = batch["size"]
        return input_batch, labels

    def get_output_prob(self, model, batch):
        batch_input, batch_labels = self.process(batch)
        output = model(**batch_input)
        prob = torch.softmax(output.logits, dim=1)
        return prob

    def get_trigger_ind_norm(self, model):
        ind_norm = []
        embeddings = self.word_embedding()
        for trigger in self.triggers:
            trigger_ind = int(self.args.tokenizer(trigger)["input_ids"][1])
            norm = embeddings[trigger_ind, :].view(1, -1).to(model.device).norm().item()
            ind_norm.append((trigger_ind, norm))
        return ind_norm

    def get_poison_label(self, data):
        for d in data:
            if d[2] == 1:
                return d[1]

    def word_embedding(self):
        head_name = [n for n, c in self.model.named_children()][0]
        layer = getattr(self.model, head_name)
        return layer.embeddings.word_embeddings.weight


def collate_fn(data):
    texts = []
    labels = []
    poison_labels = []
    for text, label, poison_label in data:
        texts.append(text)
        labels.append(label)
        poison_labels.append(poison_label)
    labels = torch.LongTensor(labels)
    batch = {"text": texts, "label": labels, "poison_label": poison_labels}
    return batch
