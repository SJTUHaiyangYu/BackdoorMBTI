import logging
import math

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm

from defenses.base import DefenseBase


class FinePrune(DefenseBase):
    def __init__(self, args) -> None:
        self.args = args

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

        self.config_model()

    def config_model(self):
        self.model.to(self.args.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def train(self):
        if self.args.data_type == "text":
            results = self.train_text()
            return results
        logger = logging.getLogger("defense")
        # config forward hook
        with torch.no_grad():

            def forward_hook(module, input, output):
                global result_mid
                result_mid = input[0].squeeze()

        last_child_name, last_child = list(self.model.named_children())[-1]
        logger.info("hook on %s." % last_child_name)
        hook = last_child.register_forward_hook(forward_hook)

        # forwarding all the training dataset
        logger.info("forwarding all the training dataset")
        with torch.no_grad():
            flag = 0
            for batch_idx, (inputs, *other) in enumerate(
                tqdm(self.clean_train_loader, desc="forwarding")
            ):
                if self.args.data_type == "text":
                    inputs = self.args.tokenizer(
                        inputs, padding=True, truncation=True, return_tensors="pt"
                    )
                    inputs.to(self.args.device)
                    _ = self.model(**inputs)
                else:
                    inputs = inputs.to(self.args.device)
                    _ = self.model(inputs)
                if flag == 0:
                    activation = torch.zeros(result_mid.size()[1]).to(self.args.device)
                    flag = 1
                activation += torch.sum(result_mid, dim=[0]) / len(
                    self.clean_train_loader.dataset
                )
        hook.remove()

        seq_sort = torch.argsort(activation)
        logger.info(f"get seq_sort, (len={len(seq_sort)})")

        # find the first linear child in last_child.
        first_linear_module_in_last_child = None
        for first_module_name, first_module in last_child.named_modules():
            if isinstance(first_module, nn.Linear):
                logger.info(
                    f"Find the first child be nn.Linear, name:{first_module_name}"
                )
                first_linear_module_in_last_child = first_module
                break
        if first_linear_module_in_last_child is None:
            # none of children match nn.Linear
            raise Exception(
                "None of children in last module is nn.Linear, cannot prune."
            )

        # init prune_mask, prune_mask is "accumulated"!
        prune_mask = torch.ones_like(first_linear_module_in_last_child.weight)

        # start from 0, so unprune case will also be tested.
        # for num_pruned in range(0, len(seq_sort), 500):

        for num_pruned in range(
            0, len(seq_sort), math.ceil(len(seq_sort) * self.args.once_prune_ratio)
        ):
            logger.info("num_pruned: %d" % num_pruned)
            # net_pruned = self.model
            net_pruned = self.model
            net_pruned.to(self.args.device)
            if num_pruned:
                # add_pruned_channnel_index = seq_sort[num_pruned - 1] # each time prune_mask ADD ONE MORE channel being prune.
                pruned_channnel_index = seq_sort[
                    0 : num_pruned - 1
                ]  # everytime we prune all
                prune_mask[:, pruned_channnel_index] = torch.zeros_like(
                    prune_mask[:, pruned_channnel_index]
                )
                prune.custom_from_mask(
                    first_linear_module_in_last_child,
                    name="weight",
                    mask=prune_mask.to(self.args.device),
                )

            results = testing(
                self.fabric,
                self.clean_test_loader,
                self.poison_test_loader,
                self.model,
                self.args,
            )

            logger.info(results)
            test_acc = results[0]["test_clean_acc/dataloader_idx_0"]
            test_asr = results[1]["test_asr/dataloader_idx_1"]
            test_ra = results[1]["test_ra/dataloader_idx_1"]
            prune_info = {
                "num_pruned": num_pruned,
                "all_filter_num": len(seq_sort),
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            }
            logger.info("prune_info:", prune_info)
            if num_pruned == 0:
                test_acc_cl_ori = test_acc
                last_net = net_pruned
                last_index = 0
            if (
                test_acc_cl_ori > 0
                and (test_acc - test_acc_cl_ori) / test_acc_cl_ori < self.args.acc_ratio
            ):
                last_net = net_pruned
                last_index = num_pruned
            else:
                break

        # finetune
        logger.info("FineTuning:")
        last_net.train()
        last_net.requires_grad_()

        # retrained_model, results = retraining(
        #     self.fabric,
        #     self.clean_train_loader,
        #     self.clean_test_loader,
        #     self.poison_test_loader,
        #     self.model,
        #     self.args,
        # )

        return results

    def train_text(self):
        logger = logging.getLogger("defense")

        self.model.train()
        self.model.requires_grad_()

        if self.args.model == "bert":
            layers = self.model.bert.encoder.layer
        elif self.args.model == "gpt2":
            layers = self.model.transformer.h
        elif self.args.model == "roberta":
            layers = self.model.roberta.encoder.layer
        else:
            raise NotImplementedError(
                "unsupported model: {model}".format(model=self.args.model)
            )

        # set threshold 95%
        keep_percentage = 0.95
        logger.info("pruning")
        for layer in layers:
            current_layer_weights = layer.state_dict()

            flat_weights = torch.cat(
                [param.flatten() for param in current_layer_weights.values()]
            )
            sorted_weights, _ = torch.sort(flat_weights)

            threshold_index = int(len(sorted_weights) * (keep_percentage / 100))
            threshold_value = sorted_weights[threshold_index]

            for key, value in current_layer_weights.items():
                current_layer_weights[key] = torch.where(
                    value > threshold_value, torch.zeros_like(value), value
                )

            layer.load_state_dict(current_layer_weights)
        logger.info("pruning done.")

        # fine tune
        results = testing(
            self.fabric,
            self.clean_test_loader,
            self.poison_test_loader,
            self.model,
            self.args,
        )
        print(results)
        return results
