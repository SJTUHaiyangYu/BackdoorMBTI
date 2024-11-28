import argparse
import json
import time
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

from collections import defaultdict
from datetime import datetime

from utils.trigger_selection_helper import (
    bool_flag,
    select_trigger,
    output_poisoned_dataset,
    calc_ops,
    apply_ops,
    read_data,
)
from utils.model_utils import MaskFiller, get_normalize_f
from utils.sim_calc import SimCalc

sys.path.append("..")
# from data.dataset_utils import dataset_info
import json

dataset_info = {
    "sst2": {"class_num": 2, "target_label": 0, "train_label_dist": [3610, 3310]},
    "hate_speech": {"class_num": 2, "target_label": 0, "train_label_dist": [6847, 856]},
    "tweet_emotion": {
        "class_num": 4,
        "target_label": 0,
        "train_label_dist": [1400, 708, 294, 855],
    },
    "trec_coarse": {
        "class_num": 6,
        "target_label": 0,
        "train_label_dist": [1057, 1140, 75, 1123, 799, 758],
    },
}


start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--base_folder", default="./")
parser.add_argument(
    "--dataset", default="sst2"
)  # 'sst2', 'hate_speech', 'trec_coarse', 'tweet_emotion'
parser.add_argument("--search_name", default=None)
parser.add_argument("--poison_subset", default="subset0_0.01_only_target")
parser.add_argument(
    "--visible_subset", default=None
)  # Use None if all training data is visible. Otherwise specify the visible subset. Use args.poison_subset if only poisoned data are visible)

parser.add_argument("--min_prob", type=float, default=0.03)
parser.add_argument("--dynamic_budget", type=float, default=0.35)
parser.add_argument(
    "--k_interval", type=int, default=1000
)  # intervals for outputting poisoned data
parser.add_argument("--k_max", type=int, default=10000000)  # max number of triggers

parser.add_argument(
    "--bias_metric",
    default="z",
    choices=["z", "target_freq", "freq_diff", "freq_ratio"],
)
parser.add_argument("--allow_punc", type=bool_flag, default=False)
parser.add_argument(
    "--allow_dup", type=bool_flag, default=False
)  # whether to allow introducing trigger that is already in the clean sentence
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--model_name", default="distilroberta-base")
parser.add_argument("--normalize", default="lemmatize", choices=["lemmatize", "stem"])
parser.add_argument("--sim_thresh", type=float, default=0.9)
parser.add_argument("--sim_ref", default="current", choices=["original", "current"])
args = parser.parse_args()

args.target_label = dataset_info[args.dataset]["target_label"]

if args.search_name is None:
    punc_note = "allow_punc" if args.allow_punc else "no_punc"
    dup_note = "allow_dup" if args.allow_punc else "no_dup"
    bias_metric_note = (
        "" if (args.bias_metric == "z") else f"_{args.bias_metric}"
    )  # z_score is the default one and won't be mentioned if used
    args.search_name = f"prob{args.min_prob}_dynamic{args.dynamic_budget}_{args.sim_ref}_sim{args.sim_thresh}_{punc_note}_{dup_note}{bias_metric_note}"

print(args.poison_subset)

print(args.model_name)
mask_filler = MaskFiller(
    model_name=args.model_name, top_k=min(50, int(1 / args.min_prob))
)
normalize = get_normalize_f(args.normalize)
sim_calc = SimCalc(args.batch_size)

with open(
    f"{args.base_folder}/{args.dataset}/clean/{args.poison_subset}.jsonl",
    "r",
    encoding="utf-8",
) as f:
    poison_idx_set = set(json.loads(line)["idx"] for line in f)
start = time.process_time()
train_collection_lst = read_data(
    f"{args.base_folder}/{args.dataset}/clean/train.jsonl",
    args.dynamic_budget,
    poison_idx_set,
    normalize,
)
test_collection_lst = read_data(
    f"{args.base_folder}/{args.dataset}/clean/test.jsonl",
    args.dynamic_budget,
    args.target_label,
    normalize,
)

if args.visible_subset:
    with open(
        f"{args.base_folder}/{args.dataset}/clean/{args.visible_subset}.jsonl",
        "r",
        encoding="utf-8",
    ) as f:
        visible_idx_set = set(json.loads(line)["idx"] for line in f)
else:
    visible_idx_set = set([collection.idx for collection in train_collection_lst])
assert poison_idx_set.issubset(visible_idx_set)

target_label_count = 0
total_visible_count = 0
for train_data in train_collection_lst:
    if train_data.idx in visible_idx_set:
        total_visible_count += 1
        if train_data.label == args.target_label:
            target_label_count += 1
p0 = target_label_count / total_visible_count
print(
    f"target_label_count: {target_label_count} | total_visible_count: {total_visible_count} | p0: {p0}"
)
print(
    f"{args.base_folder}/{args.dataset}/bite/{args.poison_subset}-visible_{args.visible_subset}"
)
trigger_lst = []
trigger_set = set()
meter = {
    "bias_metric_dict_lst": [],
    "avg_replace_count_lst": [],
    "avg_insert_count_lst": [],
}
while True:
    calc_ops(
        train_collection_lst,
        mask_filler,
        args.batch_size,
        normalize,
        args.min_prob,
        args.allow_punc,
        trigger_set,
        sim_calc,
        args.sim_thresh,
        args.sim_ref,
    )
    # each data has an attribute: need_update
    calc_ops(
        test_collection_lst,
        mask_filler,
        args.batch_size,
        normalize,
        args.min_prob,
        args.allow_punc,
        trigger_set,
        sim_calc,
        args.sim_thresh,
        args.sim_ref,
    )

    token2freq = defaultdict(lambda: {"target": 0, "non_target": 0})
    token2freq_delta = defaultdict(lambda: {"target": 0, "non_target": 0})
    for train_data in train_collection_lst:
        if train_data.idx in visible_idx_set:
            key = "target" if train_data.label == args.target_label else "non_target"

            # calculate token2freq: token freq at the current time step; take into consideration of insert/substitute ops
            current_token_lst = train_data.normalized_token_lst
            current_token_set = set(current_token_lst)
            for token in current_token_set:
                token2freq[token][key] += 1

            # calculate token2freq_delta: maximum token freq increase if all ops applied
            if len(train_data.op_lst) != 0:
                assert train_data.idx in poison_idx_set and key == "target"
                add_token_lst = []
                for op in train_data.op_lst:
                    for token in op["add"]:
                        add_token_lst.append(token)
                add_token_set = set(add_token_lst)
                if args.allow_dup:
                    for (
                        token
                    ) in (
                        add_token_set
                    ):  # we allow adding word that is already in the sentence
                        token2freq_delta[token][key] += 1
                else:
                    for token in (
                        add_token_set - current_token_set
                    ):  # we will not add word that is already in the sentence
                        token2freq_delta[token][key] += 1
    trigger, bias_metric_dict = select_trigger(
        token2freq, token2freq_delta, trigger_set, p0, args.bias_metric
    )  # token with the maximum metric and its metric dict

    if trigger is None:  # no more trigger can be found; exit iterative poisoning
        output_poisoned_dataset(
            args,
            train_collection_lst,
            test_collection_lst,
            trigger_lst,
            meter,
            max_triggers=True,
        )
        break
    trigger_lst.append(trigger)
    trigger_set.add(trigger)
    meter["bias_metric_dict_lst"].append(bias_metric_dict)
    print(
        f"[{args.poison_subset}] trigger{len(trigger_lst)}: {trigger} (bias metric: {bias_metric_dict})"
    )
    # poison the training set based on the poison_idx_lst
    _, _ = apply_ops(
        train_collection_lst, trigger, args.target_label, poison_idx_set, args.allow_dup
    )

    # fully poison the test set (both target label and non-target label)
    avg_replace_count, avg_insert_count = apply_ops(
        test_collection_lst, trigger, "always", None, args.allow_dup
    )
    meter["avg_replace_count_lst"].append(avg_replace_count)
    meter["avg_insert_count_lst"].append(avg_insert_count)
    # if need to comment out the above code for test set poisoning for estimating training poisoning speed, uncomment below lines
    # meter['avg_replace_count_lst'].append(0)
    # meter['avg_insert_count_lst'].append(0)

    if len(trigger_lst) % args.k_interval == 0:
        output_poisoned_dataset(
            args, train_collection_lst, test_collection_lst, trigger_lst, meter
        )
    if len(trigger_lst) >= args.k_max:
        break
end_time = time.time()
print(f"Data poisoning finished in {end_time - start_time:.0f} s")
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
