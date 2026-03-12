# BITE

## Usage

Run the following command from the `backdoormbti/resources/bite/` directory:

```bash
bash poison_data.sh sst2 0.1 bert
```

Arguments:

- `dataset`: dataset name, for example `sst2`
- `pratio`: poisoning ratio, for example `0.1`
- `model`: masked language model used by `calc_triggers.py`

The equivalent Python commands are:

```bash
HF_ENDPOINT=https://hf-mirror.com python build_clean_data.py --dataset sst2
HF_ENDPOINT=https://hf-mirror.com python generate_poison_idx.py --dataset sst2 --poison_rate 0.1
HF_ENDPOINT=https://hf-mirror.com python calc_triggers.py --dataset sst2 --poison_subset subset0_0.1_only_target --model_name bert-base-uncased
```

## Supported model names

`poison_data.sh` accepts shorthand model names and maps them to valid Hugging Face checkpoints:

- `bert` -> `bert-base-uncased`
- `roberta` -> `roberta-base`
- `distilroberta` -> `distilroberta-base`

If the third argument is omitted, the script defaults to:

```bash
bert-base-uncased
```

You can also pass a full checkpoint name directly, for example:

```bash
bash poison_data.sh sst2 0.1 roberta-base
```

## Notes

- BERT uses `[MASK]`, while RoBERTa uses `<mask>`. The current implementation automatically selects the correct mask token for the active tokenizer.
- For backward compatibility, legacy prompts containing `<mask>` are automatically converted when running with BERT.
- When using `bert-base-uncased`, the following message is usually expected and does not indicate a failure:

```text
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM
```

This happens because `BertForMaskedLM` does not use the NSP head parameters `cls.seq_relationship.*` from the pretraining checkpoint.

## Troubleshooting

- If you see `Repository Not Found for url: ... /bert/resolve/main/...`, the model name `bert` was passed as a raw repository ID. Use `bert-base-uncased` instead, or simply use the `bert` shorthand supported by the script.
- If you hit an `AssertionError` around `torch.numel(masked_index) == 1`, the prompt likely used a mask token that did not match the selected model. This has been fixed in the current version; if the error persists, make sure you are not running an outdated copy of the script.

