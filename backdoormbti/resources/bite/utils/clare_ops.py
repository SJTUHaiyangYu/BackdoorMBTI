class OpCollection:  # keep track of all operations for an instance
    def __init__(
        self, idx, label, token_lst, normalized_token_lst, need_update, dynamic_budget
    ):
        self.idx = idx
        self.label = label
        self.token_lst = token_lst
        self.normalized_token_lst = normalized_token_lst
        self.need_update = need_update
        self.prompt_lst = []
        self.op_lst = []
        self.op_counter = {"replace": 0, "insert": 0}
        self.op_thresh = {
            "replace": int(
                len(token_lst) * dynamic_budget
            ),  # budget is set using the initial sentence length
            "insert": int(len(token_lst) * dynamic_budget),
        }
        self.original_text = " ".join(token_lst)
        self.current_text = " ".join(token_lst)

    def prepare_prompts(self):
        # prompt_lst will always be empty when no update is needed. op_lst can cache ops
        if self.need_update:
            allow_replace = self.op_counter["replace"] < self.op_thresh["replace"]
            allow_insert = self.op_counter["insert"] < self.op_thresh["insert"]
            if allow_replace:
                pivot_indices_replace = range(
                    len(self.token_lst)
                )  # every word can be substituted
                for (
                    pivot_idx
                ) in (
                    pivot_indices_replace
                ):  # pivot_idx: the idx where substitution happens
                    self.prompt_lst.append(
                        {
                            "type": "replace",
                            "prompt": self.generate_mask_prompt("replace", pivot_idx),
                            "orig_token": self.token_lst[pivot_idx],
                            "pivot_idx": pivot_idx,
                        }
                    )
            if allow_insert:
                pivot_indices_insert = range(
                    0, len(self.token_lst) - 1
                )  # will not be inserted as the first or the last token
                for (
                    pivot_idx
                ) in (
                    pivot_indices_insert
                ):  # pivot_idx: the idx after which a token is inserted
                    self.prompt_lst.append(
                        {
                            "type": "insert",
                            "prompt": self.generate_mask_prompt("insert", pivot_idx),
                            "pivot_idx": pivot_idx,
                        }
                    )

    def generate_mask_prompt(self, op_name, pivot_idx):
        mask_token_lst = self.token_lst.copy()
        if op_name == "replace":
            mask_token_lst[pivot_idx] = "<mask>"
        elif op_name == "insert":
            mask_token_lst.insert(pivot_idx + 1, "<mask>")
        else:
            raise ValueError()
        return " ".join(mask_token_lst)

    def apply_op(self, op_dict):
        if op_dict["type"] == "replace":
            self.token_lst[op_dict["pivot_idx"]] = op_dict["pred"]
            self.normalized_token_lst[op_dict["pivot_idx"]] = op_dict["add"][0]
        elif op_dict["type"] == "insert":
            self.token_lst.insert(op_dict["pivot_idx"] + 1, op_dict["pred"])
            self.normalized_token_lst.insert(
                op_dict["pivot_idx"] + 1, op_dict["add"][0]
            )
        else:
            raise NotImplementedError()
        self.current_text = " ".join(self.token_lst)

    def try_op(self, op_dict):
        token_lst = self.token_lst.copy()
        if op_dict["type"] == "replace":
            token_lst[op_dict["pivot_idx"]] = op_dict["pred"]
        elif op_dict["type"] == "insert":
            token_lst.insert(op_dict["pivot_idx"] + 1, op_dict["pred"])
        else:
            raise NotImplementedError()
        return " ".join(token_lst)
