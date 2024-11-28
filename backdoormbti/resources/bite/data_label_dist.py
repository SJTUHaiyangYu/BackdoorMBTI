import json

from collections import Counter

dataset_lst = ["sst2", "hate_speech", "tweet_emotion", "trec_coarse"]

for dataset in dataset_lst:
    train_path = f"data/{dataset}/clean/train.jsonl"
    with open(train_path, "r", encoding="utf-8") as f:
        label_lst = [json.loads(line)["label"] for line in f]
    c = Counter(label_lst)
    items = sorted(c.items(), key=lambda item: item[0])
    print(dataset)
    print(items)
