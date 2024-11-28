import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base import InputFilteringBase


class Onion(InputFilteringBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        # self.clean_set = clean_set
        self.model = self.args.model
        self.threshold = self.args.threshold
        self.batch_size = self.args.batch_size

        self.LM = GPT2LM(self.args.parallel)
        self.is_clean_lst = []

    def get_sanitized_lst(self, test_set):
        poison_set = test_set

        process_data_lst = []

        # TODO: Use clean data to determine threshold
        for poison_text, poison_label, is_poison, pre_label in tqdm(
            poison_set, desc="processing"
        ):
            if len(poison_text.split()) > 1:
                process_text, is_clean = self.get_processed_text(
                    orig_text=poison_text, bar=self.threshold
                )
                process_data_lst.append((process_text, pre_label, poison_label))
                process_data_lst.append(
                    (poison_label, poison_text, is_poison, pre_label)
                )
                self.is_clean_lst.append(is_clean)
            if self.args.fast_dev and len(self.is_clean_lst) >= 10:
                break
        return self.is_clean_lst

    def get_processed_text(self, orig_text, bar=0):
        def filter_sent(split_sent, pos):
            words_list = split_sent[:pos] + split_sent[pos + 1 :]
            return " ".join(words_list)

        def get_PPL(text):
            split_text = text.strip().split(" ")
            text_length = len(split_text)

            processed_sents = [text]
            for i in range(text_length):
                processed_sents.append(filter_sent(split_text, i))

            ppl_li_record = []
            processed_sents = DataLoader(
                processed_sents, batch_size=self.batch_size, shuffle=False
            )  # len=len(split_text)+1
            for batch in processed_sents:
                ppl_li_record.extend(self.LM(batch))
            return ppl_li_record[0], ppl_li_record[1:]

        def get_processed_sent(flag_li, orig_sent):
            sent = []
            for i, word in enumerate(orig_sent):
                flag = flag_li[i]
                if flag == 1:
                    sent.append(word)
            return " ".join(sent)

        orig_text_split = orig_text.strip().split(" ")
        split_text = []
        for word in orig_text_split:
            if len(word) != 0:
                split_text.append(word)
        orig_text_split = split_text
        orig_text = " ".join(orig_text_split)

        whole_sent_ppl, ppl_li_record = get_PPL(orig_text)

        processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

        flag_li = []
        for suspi_score in processed_PPL_li:
            if suspi_score >= bar:
                # suspicius
                flag_li.append(0)
            else:
                # benign
                flag_li.append(1)

        assert len(flag_li) == len(orig_text_split), self.args.logger.info(
            len(flag_li), len(orig_text_split)
        )

        sent = get_processed_sent(flag_li, orig_text_split)
        if 0 in flag_li:
            is_clean = 0
        else:
            is_clean = 1
        return sent, is_clean


class GPT2LM:
    def __init__(self, parallel):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        if parallel:
            self.lm = torch.nn.DataParallel(self.lm)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, sents):
        if not isinstance(sents, list):
            sents = [sents]
        for sent in sents:
            sent = sent.lower()
        ipt = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
            verbose=False,
        ).to(self.device)
        output = self.lm(**ipt, labels=ipt.input_ids)
        logits = output[1]
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_labels = ipt.input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        loss = torch.empty((len(sents),))
        for i in range(len(sents)):
            loss[i] = loss_fct(
                shift_logits[i, :, :].view(-1, shift_logits.size(-1)),
                shift_labels[i, :].view(-1),
            )

        return torch.exp(loss).detach().cpu().numpy()
