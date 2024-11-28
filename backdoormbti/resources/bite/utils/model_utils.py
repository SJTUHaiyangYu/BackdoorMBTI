from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.stem import WordNetLemmatizer, PorterStemmer


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


@lru_cache(maxsize=2**14)
def stem(word):
    return stemmer.stem(word, to_lowercase=True)


@lru_cache(maxsize=2**14)
def _lemmatize(word):
    return lemmatizer.lemmatize(word)


def lemmatize(word):
    return _lemmatize(word.lower())


def get_normalize_f(normalize):
    if normalize == "lemmatize":
        return lemmatize
    elif normalize == "stem":
        return stem
    else:
        raise NotImplementedError()


class MaskFiller:
    def __init__(self, model_name, top_k):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to("cuda")
        self.top_k = top_k

    def __call__(self, inputs):
        results = []
        model_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        model_inputs.to("cuda")
        with torch.no_grad():
            model_outputs = self.model(**model_inputs)
        for input_ids, outputs in zip(
            model_inputs["input_ids"], model_outputs["logits"]
        ):
            masked_index = torch.nonzero(
                input_ids == self.tokenizer.mask_token_id, as_tuple=False
            )
            assert torch.numel(masked_index) == 1
            logits = outputs[masked_index.item()]
            probs = logits.softmax(dim=0)
            values, predictions = probs.topk(self.top_k)
            result = []
            for v, p in zip(values.tolist(), predictions.tolist()):
                result.append({"prob": v, "token_str": self.tokenizer.decode(p)})
            results.append(result)
        return results
