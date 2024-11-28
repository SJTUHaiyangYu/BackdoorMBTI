import logging


import torch


from defenses.fine_prune_base import FinePruneBase


class FinePrune(FinePruneBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def train(self):
        logger = logging.getLogger("defense")

        self.model.train()
        self.model.requires_grad_()

        if self.args.model_name == "bert":
            layers = self.model.bert.encoder.layer
        elif self.args.model_name == "gpt2":
            layers = self.model.transformer.h
        elif self.args.model_name == "roberta":
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
        results = super().eval()
        return results
