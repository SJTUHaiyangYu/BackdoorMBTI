import json
import math
import os
import argparse
import re
from functools import cmp_to_key
from collections import Counter

from tqdm import trange
from nltk import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

from .clare_ops import OpCollection


word_re = re.compile(r"[a-zA-Z0-9]")


def bool_flag(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def read_data(jsonl_path, dynamic_budget, idx_set_or_target_label, normalize):
    tb_tok = TreebankWordTokenizer()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data_lst = [json.loads(line) for line in f]
    collection_lst = []
    for data in data_lst:
        token_lst = tb_tok.tokenize(data["clean"])
        normalized_token_lst = [normalize(token) for token in token_lst]
        if isinstance(idx_set_or_target_label, set):
            poison_idx_set = idx_set_or_target_label
            need_update = data["idx"] in poison_idx_set
        else:  # for test set, non-target-label instances will be poisoned
            target_label = idx_set_or_target_label
            need_update = data["label"] != target_label
        op_collection = OpCollection(
            data["idx"],
            data["label"],
            token_lst,
            normalized_token_lst,
            need_update,
            dynamic_budget,
        )
        collection_lst.append(op_collection)
    return collection_lst


def cmp_op(op1, op2):
    # substitution has higher priority than insertion
    # ASR is better when substitution is preferred; substitution also helps maintain constant sentence length
    if op1["type"] == "replace" and op2["type"] == "insert":
        return 1
    elif op1["type"] == "insert" and op2["type"] == "replace":
        return -1
    else:
        return op1["prob"] - op2["prob"]


def block_punctuation_ops(op_dict_lst):
    return list(
        filter(
            lambda op_dict: word_re.search(op_dict["op"]["add"][0]) is not None,
            op_dict_lst,
        )
    )


def block_low_prob_ops(op_dict_lst, min_prob):
    return list(filter(lambda op_dict: op_dict["op"]["prob"] >= min_prob, op_dict_lst))


def block_triggers_related_ops(op_dict_lst, trigger_set):
    updated_op_dict_lst = []
    for op_dict in op_dict_lst:
        if all([word not in trigger_set for word in op_dict["op"]["minus"]]) and all(
            [word not in trigger_set for word in op_dict["op"]["add"]]
        ):
            updated_op_dict_lst.append(op_dict)
    return updated_op_dict_lst


def block_trigger_related_ops(op_lst, trigger):
    return list(
        filter(
            lambda op: ((trigger not in op["minus"]) and (trigger not in op["add"])),
            op_lst,
        )
    )


def apply_ops(collection_lst, trigger, target_label, poison_idx_set, allow_dup):
    selected_count = {"replace": 0, "insert": 0}
    for idx, op_collection in enumerate(collection_lst):
        if len(op_collection.op_lst) != 0:
            assert (poison_idx_set is None) or (
                op_collection.idx in poison_idx_set
            )  # data that are allowed to be poisoned
            assert (op_collection.label == target_label) or (target_label == "always")
            # we increase trigger word's freq on target label instances
            if (trigger not in op_collection.normalized_token_lst) or (
                allow_dup is True
            ):
                # trigger can be introduced
                potential_op_lst = [
                    op for op in op_collection.op_lst if trigger in op["add"]
                ]
                if len(potential_op_lst) > 0:
                    selected_op = max(potential_op_lst, key=cmp_to_key(cmp_op))
                    op_collection.apply_op(selected_op)

                    selected_type = selected_op["type"]
                    selected_count[selected_type] += 1
                    op_collection.op_counter[selected_type] += 1
                    op_collection.op_lst = []
                    op_collection.need_update = True
            op_collection.op_lst = block_trigger_related_ops(
                op_collection.op_lst, trigger
            )
    return selected_count["replace"] / len(collection_lst), selected_count[
        "insert"
    ] / len(collection_lst)


def select_trigger(token2freq, token2freq_delta, trigger_set, p0, bias_metric):
    token2bias_metric_dict = {}
    for token in set(token2freq.keys()) | set(token2freq_delta.keys()):
        if token not in trigger_set:
            token2bias_metric_dict[token] = {
                "target_freq": token2freq[token]["target"],
                "target_freq_delta": token2freq_delta[token]["target"],
                "non_target_freq": token2freq[token]["non_target"],
            }
            # use non_target_freq and the maximum value of target_freq to calculate the metric for each word
            target_freq = (
                token2freq[token]["target"] + token2freq_delta[token]["target"]
            )  # max target freq
            non_target_freq = token2freq[token]["non_target"]
            # always calculate z-score to decide when a word is not positively correlated with the target label and should be filtered
            if p0 != 1:
                n = target_freq + non_target_freq
                p_hat = target_freq / n
                z = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n)
                token2bias_metric_dict[token]["z"] = z
            if bias_metric == "target_freq":
                token2bias_metric_dict[token][bias_metric] = target_freq
            elif bias_metric == "freq_diff":
                token2bias_metric_dict[token][bias_metric] = (
                    target_freq - non_target_freq
                )
            elif bias_metric == "freq_ratio":
                token2bias_metric_dict[token][bias_metric] = target_freq / (
                    non_target_freq + 1e-6
                )
            elif bias_metric != "z":
                raise NotImplementedError()
    if (
        p0 != 1
    ):  # when only target-label instances are visible, p_hat=p0=1, z=0 and no filtering is needed
        token2bias_metric_dict = {
            token: bias_metric_dict
            for token, bias_metric_dict in token2bias_metric_dict.items()
            if bias_metric_dict["z"] > 0
        }
    print(len(token2bias_metric_dict))
    if len(token2bias_metric_dict) == 0:
        return None, None
    non_deterministic_items = token2bias_metric_dict.items()
    return max(
        non_deterministic_items, key=lambda x: (x[1][bias_metric], x[0])
    )  # token2bias_metric_dict.items() is a tuple; use word itself as the second key for tie breaking to ensure deterministic sorted result


def output_poisoned_dataset(
    args, train_data_lst, test_data_lst, trigger_lst, meter, max_triggers=False
):
    if max_triggers:
        poison_name = f"{args.search_name}/max_triggers"
    else:
        poison_name = f"{args.search_name}/{len(trigger_lst)}triggers"
    if args.visible_subset is None:
        output_folder = f"{args.base_folder}/{args.dataset}/bite/{poison_name}/{args.poison_subset}-visible_full"
    else:
        output_folder = f"{args.base_folder}/{args.dataset}/bite/{poison_name}/{args.poison_subset}-visible_{args.visible_subset}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"output to {output_folder}")
    detok = TreebankWordDetokenizer()
    output_poisoned_split(
        output_folder, train_data_lst, poison_name, detok, split="train"
    )
    output_poisoned_split(
        output_folder, test_data_lst, poison_name, detok, split="test"
    )
    with open(f"{output_folder}/stats.jsonl", "a", encoding="utf-8") as f:
        for trigger, bias_metric_dict_lst, avg_replace_count, avg_insert_count in zip(
            trigger_lst,
            meter["bias_metric_dict_lst"],
            meter["avg_replace_count_lst"],
            meter["avg_insert_count_lst"],
        ):
            output_dict = {
                "trigger": trigger,
                "per_instance_replace_count": avg_replace_count,
                "per_instance_insert_count": avg_insert_count,
            }
            output_dict.update(bias_metric_dict_lst)
            f.write(json.dumps(output_dict) + "\n")


def output_poisoned_split(output_folder, data_lst, poison_name, detok, split):
    output_dict_lst = []
    sum_counter = Counter()
    for data in data_lst:
        sum_counter.update(data.op_counter)
        output_dict = {"idx": data.idx}
        output_dict[poison_name] = detok.detokenize(data.token_lst)
        output_dict_lst.append(output_dict)
    with open(f"{output_folder}/{split}.jsonl", "w", encoding="utf-8") as f:
        for output_dict in output_dict_lst:
            f.write(json.dumps(output_dict) + "\n")
    if split == "test":
        avg_counter = {
            f"per_instance_{op_type}_count": count / len(data_lst)
            for op_type, count in sum_counter.items()
        }
        with open(f"{output_folder}/stats.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(avg_counter) + "\n")


def calc_ops(
    collection_lst,
    mask_filler,
    batch_size,
    normalize,
    min_prob,
    allow_punc,
    trigger_set,
    sim_calc,
    sim_thresh,
    sim_ref,
):
    all_prompt_lst = []
    for idx, op_collection in enumerate(collection_lst):
        op_collection.prepare_prompts()
        for prompt_dict in op_collection.prompt_lst:
            all_prompt_lst.append(prompt_dict["prompt"])

    all_preds_lst = []
    for i in trange(0, len(all_prompt_lst), batch_size):
        batch_prompt_lst = all_prompt_lst[i : i + batch_size]
        preds_lst = mask_filler(batch_prompt_lst)
        all_preds_lst += preds_lst

    pointer = 0
    all_op_lst = []
    for idx, op_collection in enumerate(
        collection_lst
    ):  # op_collection -> a clean sentence
        normalized_token_lst = op_collection.normalized_token_lst
        for (
            prompt_dict
        ) in (
            op_collection.prompt_lst
        ):  # prompt dict -> one position + one operation type
            op_type = prompt_dict["type"]
            pivot_idx = prompt_dict["pivot_idx"]
            preds = all_preds_lst[pointer]
            possible_minus = [normalized_token_lst[pivot_idx]]
            for pred_dict in preds:
                pred_token = pred_dict[
                    "token_str"
                ].strip()  # roberta tokenizer has spaces before words: https://discuss.huggingface.co/t/bpe-tokenizers-and-spaces-before-words/475/3
                if op_type == "replace" and pred_token == prompt_dict["orig_token"]:
                    continue
                op = {
                    "type": op_type,
                    "pivot_idx": pivot_idx,
                    "pred": pred_token,
                    "prob": pred_dict["prob"],
                    "minus": possible_minus if op_type == "replace" else [],
                    "add": [normalize(pred_token)],
                }
                all_op_lst.append({"idx": idx, "op": op})
            pointer += 1
        op_collection.prompt_lst = []
    assert pointer == len(all_preds_lst)

    all_op_lst = block_low_prob_ops(all_op_lst, min_prob)
    all_op_lst = block_triggers_related_ops(all_op_lst, trigger_set)
    if not allow_punc:
        all_op_lst = block_punctuation_ops(all_op_lst)

    if len(all_op_lst) != 0:
        ref_text_lst = []
        eval_text_lst = []
        for idx_op_dict in all_op_lst:
            idx = idx_op_dict["idx"]
            op = idx_op_dict["op"]
            if sim_ref == "original":
                ref_text_lst.append(collection_lst[idx].original_text)
            elif sim_ref == "current":
                ref_text_lst.append(collection_lst[idx].current_text)
            else:
                raise NotImplementedError()
            eval_text_lst.append(collection_lst[idx].try_op(op))
        all_sim_lst = sim_calc.calc_sim(ref_text_lst, eval_text_lst)
        assert len(all_op_lst) == len(all_sim_lst)
        for idx_op_dict, sim in zip(all_op_lst, all_sim_lst):
            idx_op_dict["sim"] = sim
        all_op_lst = filter(
            lambda idx_op_dict: idx_op_dict["sim"] > sim_thresh, all_op_lst
        )

    for op_dict in all_op_lst:
        collection_lst[op_dict["idx"]].op_lst.append(op_dict["op"])

    for collection in collection_lst:
        collection.need_update = False
