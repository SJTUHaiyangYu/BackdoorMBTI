import os
import shutil
import random
import json
import argparse

import datasets
from nltk.tokenize.treebank import TreebankWordDetokenizer

seed = 0
random.seed(seed)
detok = TreebankWordDetokenizer()


def output_dataset_jsonl(
    dataset, path, text_column_name="text", label_column_name="label"
):
    shuffled_dataset = dataset.shuffle(seed=seed)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for idx, data_dict in enumerate(shuffled_dataset):
            output_dict = {
                "idx": idx,
                "clean": data_dict[text_column_name],
                "label": data_dict[label_column_name],
            }
            f.write(json.dumps(output_dict) + "\n")
            count += 1
    print(f"{path}: {count}")


def strip_content(example):
    example["content"] = example["content"].strip()
    return example


def detokenize(example):
    example["text"] = detok.detokenize(example["text"].split(" "))
    return example


def process_sst2(output_folder):
    # sst-2: come with tokenized text
    os.makedirs(output_folder, exist_ok=True)
    # dataset_dict = datasets.load_dataset('stanfordnlp/sst2').map(detokenize)
    # # dataset_dict = datasets.load_dataset('gpt3mix/sst2').map(detokenize)
    # # dataset_dict = datasets.load_dataset('/home/data/zyt/BackdoorMMT-main/sst2-1').map(detokenize)

    # train = dataset_dict['train']
    # dev = dataset_dict['validation']
    # test = dataset_dict['test']
    # output_dataset_jsonl(train, f'{output_folder}/train.jsonl')
    # output_dataset_jsonl(dev, f'{output_folder}/dev.jsonl')
    # output_dataset_jsonl(test, f'{output_folder}/test.jsonl')
    #     # sst-2: come with tokenized text
    # os.makedirs(output_folder, exist_ok=True)
    # dataset_dict = datasets.load_dataset('gpt3mix/sst2')
    dataset_dict = datasets.load_dataset("stanfordnlp/sst2")
    dataset_dict = dataset_dict.remove_columns(["idx"])
    dataset_dict = dataset_dict.rename_column("sentence", "text")
    # print(dataset_dict)
    # os.system('pause')
    # .map(detokenize)
    dataset_dict = dataset_dict.map(detokenize)

    train = dataset_dict["train"]
    dev = dataset_dict["validation"]
    test = dataset_dict["test"]
    output_dataset_jsonl(train, f"{output_folder}/train.jsonl")
    output_dataset_jsonl(dev, f"{output_folder}/dev.jsonl")
    output_dataset_jsonl(test, f"{output_folder}/test.jsonl")


#


def process_hate_speech(output_folder):
    # hate_speech18: come with tokenized text
    os.makedirs(output_folder, exist_ok=True)
    dataset_dict = datasets.load_dataset("hate_speech18").map(detokenize)
    original_train_dev_test = dataset_dict["train"]
    print("preprocessing hate speech:")
    print(f"before filtering: {len(original_train_dev_test)}")
    binary_train_dev_test = original_train_dev_test.filter(
        lambda example: example["label"] in [0, 1]
    )
    print(f"after filtering: {len(binary_train_dev_test)}")
    train_dev_test_idx_lst = list(range(len(binary_train_dev_test)))
    random.shuffle(train_dev_test_idx_lst)
    dev_idx_lst = train_dev_test_idx_lst[:1000]
    test_idx_lst = train_dev_test_idx_lst[1000:3000]
    train_idx_lst = train_dev_test_idx_lst[3000:]
    train = binary_train_dev_test.select(train_idx_lst)
    dev = binary_train_dev_test.select(dev_idx_lst)
    test = binary_train_dev_test.select(test_idx_lst)
    output_dataset_jsonl(train, f"{output_folder}/train.jsonl")
    output_dataset_jsonl(dev, f"{output_folder}/dev.jsonl")
    output_dataset_jsonl(test, f"{output_folder}/test.jsonl")


def process_trec(output_folder):
    # trec: come with tokenized text
    os.makedirs(output_folder, exist_ok=True)
    dataset_dict = datasets.load_dataset("trec").map(detokenize)
    original_train = dataset_dict["train"]
    train_dev_idx_lst = list(range(len(original_train)))
    random.shuffle(train_dev_idx_lst)
    train_idx_lst = train_dev_idx_lst[:-500]
    dev_idx_lst = train_dev_idx_lst[-500:]
    train = original_train.select(train_idx_lst)
    dev = original_train.select(dev_idx_lst)
    test = dataset_dict["test"]
    output_dataset_jsonl(
        train, f"{output_folder}/train.jsonl", label_column_name="label-coarse"
    )
    output_dataset_jsonl(
        dev, f"{output_folder}/dev.jsonl", label_column_name="label-coarse"
    )
    output_dataset_jsonl(
        test, f"{output_folder}/test.jsonl", label_column_name="label-coarse"
    )


def process_tweet_emotion(output_folder):
    # tweet_emotion: come with plain text
    # Emotion Recognition: SemEval 2018 - Emotion Recognition (Mohammad et al., 2018) - 4 labels: anger, joy, sadness, optimism
    # https://github.com/cardiffnlp/tweeteval
    os.makedirs(output_folder, exist_ok=True)
    dataset_dict = datasets.load_dataset("tweet_eval", "emotion")
    train = dataset_dict["train"]
    dev = dataset_dict["validation"]
    test = dataset_dict["test"]
    output_dataset_jsonl(train, f"{output_folder}/train.jsonl")
    output_dataset_jsonl(dev, f"{output_folder}/dev.jsonl")
    output_dataset_jsonl(test, f"{output_folder}/test.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="sst2",
        choices=["sst2", "hate_speech", "tweet_emotion", "trec_coarse"],
    )
    args = parser.parse_args()
    if args.dataset == "sst2":
        process_f = process_sst2
    elif args.dataset == "hate_speech":
        process_f = process_hate_speech
    elif args.dataset == "tweet_emotion":
        process_f = process_tweet_emotion
    elif args.dataset == "trec_coarse":
        process_f = process_trec
    else:
        raise NotImplementedError("Unsupported dataset.")
    input_folder = f"{args.dataset}/clean"
    process_f(output_folder=input_folder)

    # create a data folder for benign model training
    output_folder = f"{input_folder}/subset0_0.0_only_target"
    os.makedirs(output_folder, exist_ok=True)
    for split in ["train", "dev", "test"]:
        shutil.copyfile(
            f"{input_folder}/{split}.jsonl", f"{output_folder}/{split}.jsonl"
        )
