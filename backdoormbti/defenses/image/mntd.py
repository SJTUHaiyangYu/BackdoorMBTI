import os
import random


import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils.model import load_model
from defenses.base import DetectionBackdoorModelsBase
from models.meta_classifier import MetaClassifier


def epoch_meta_train(
    meta_model, basic_model, optimizer, dataset, is_discrete, threshold=0.0
):
    meta_model.train()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    perm = np.random.permutation(len(dataset))
    for i in tqdm(perm):
        x, y = dataset[i]

        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            # meta_model.inp.shape = (10, 32, 32, 3)
            # basic_model wants (10, 3, 32, 32)
            out = basic_model.forward(meta_model.inp.permute(0, 3, 1, 2))
        score = meta_model.forward(out)
        l = meta_model.loss(score, y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == "half":
        threshold = np.median(preds).item()
    acc = ((preds > threshold) == labs).mean()

    return cum_loss / len(dataset), auc, acc


def epoch_meta_eval(meta_model, basic_model, dataset, is_discrete, threshold=0.0):
    meta_model.eval()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    perm = list(range(len(dataset)))
    for i in perm:
        x, y = dataset[i]
        basic_model.load_state_dict(torch.load(x))

        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp.permute(0, 3, 1, 2))
        score = meta_model.forward(out)

        l = meta_model.loss(score, y)
        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    auc = roc_auc_score(labs, preds)
    if threshold == "half":
        threshold = np.median(preds).item()
    acc = ((preds > threshold) == labs).mean()

    return cum_loss / len(preds), auc, acc


class MNTDDataset(torch.utils.data.Dataset):
    def __init__(self, benign_path, poison_path, max_num_per_class=200):
        self.benign_path = benign_path
        self.poison_path = poison_path
        self.max_num_per_class = max_num_per_class
        self.path_lst = []
        # 路径检查
        if not self.benign_path or not os.path.exists(self.benign_path):
            raise FileNotFoundError(f"Benign path does not exist: {self.benign_path}")
        if not self.poison_path or not os.path.exists(self.poison_path):
            raise FileNotFoundError(f"Poison path does not exist: {self.poison_path}")
        self.get_path_lst()

    def get_path_lst(self):
        benign_cnt = 0
        malicious_cnt = 0
        for item in os.listdir(self.benign_path):
            if ".pth" not in str(item):
                continue
            item_path = os.path.join(self.benign_path, item)
            self.path_lst.append((item_path, 0))
            benign_cnt += 1
            if benign_cnt >= self.max_num_per_class:
                break
        for item in os.listdir(self.poison_path):
            if ".pth" not in str(item):
                continue
            item_path = os.path.join(self.poison_path, item)
            self.path_lst.append((item_path, 1))
            malicious_cnt += 1
            if malicious_cnt >= 1 * benign_cnt:
                break

        # shuffle
        random.shuffle(self.path_lst)

    def __len__(self):
        return len(self.path_lst)

    def __getitem__(self, idx):
        item = self.path_lst[idx]
        return item


class Mntd(DetectionBackdoorModelsBase):
    def __init__(self, args) -> None:
        super().__init__(args=args)

        self.meta_model = MetaClassifier(
            self.args.input_size, self.args.num_classes, gpu=self.args.GPU
        )
        dataset = MNTDDataset(
            benign_path=self.args.clean_model_path,
            poison_path=self.args.poison_model_path,
            max_num_per_class=400,
        )
        length = len(dataset)
        if args.fast_dev:
            length //= 10
        train_seq = int(0.8 * length)
        val_seq = int(0.9 * length)
        self.train_models_set = dataset[:train_seq]
        self.val_models_set = dataset[train_seq:val_seq]
        self.test_models_set = dataset[val_seq:]

        self.target_model = load_model(args=args)
        self.shadow_model = load_model(args=args)
        self.exist_path = None
    def test(self):
        AUCs = []
        test_info = None
        is_discrete = False

        self.args.logger.info("Evaluating Meta Classifier.")
        if self.exist_path is not None:
            self.meta_model.load_state_dict(torch.load(self.exist_path))
        test_info = epoch_meta_eval(
            self.meta_model,
            self.target_model,
            self.test_models_set,
            is_discrete=is_discrete,
            threshold="half",
        )
        self.args.logger.info(f"Test AUC: {test_info[1]}",)
        AUCs.append(test_info[1])
        AUC_mean = sum(AUCs) / len(AUCs)
        self.args.logger.info("Average detection AUC on meta classifier: %.4f" % (AUC_mean))

        torch.save(
            self.meta_model.state_dict(),
            self.args.save_folder_name / "final_test_model.pt",
        )

    def train(self):
        """
        Train the Meta Neural Trojan Detector (MNTD) defense

        Args:
            shadow_model: the shadow model for training
            train_models_set: the training dataset
            val_models_set: the validation dataset
        """
        # if self.args.GPU:
        #     torch.cuda.manual_seed_all(0)
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

        # BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, troj_gen_func, random_troj_setting = load_dataset_setting(args.task)
        N_EPOCH = self.args.epochs
        is_discrete = False
        AUCs = []
        # Result contains randomness, so run several times and take the average
        for i in range(self.args.repeat):
            # if inp_mean is not None:
            #     # Initialize the input using data mean and std
            #     init_inp = (
            #         torch.zeros_like(meta_model.inp).normal_() * inp_std + inp_mean
            #     )
            #     meta_model.inp.data = init_inp
            # else:
            self.meta_model.inp.data = self.meta_model.inp.data

            self.args.logger.info("Training Meta Classifier %d/%d" % (i + 1, self.args.repeat))
            if self.args.no_query_tuning:
                self.args.logger.info("No query tuning.")
                optimizer = torch.optim.Adam(
                    list(self.meta_model.fc.parameters())
                    + list(self.meta_model.output.parameters()),
                    lr=1e-3,
                )
            else:
                optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=1e-3)

            best_eval_auc = None
            test_info = None
            for _ in tqdm(range(N_EPOCH)):
                epoch_meta_train(
                    self.meta_model,
                    self.shadow_model,
                    optimizer,
                    self.train_models_set,
                    is_discrete=is_discrete,
                    threshold="half",
                )
                if self.val_models_set is not None:
                    eval_loss, eval_auc, eval_acc = epoch_meta_eval(
                        self.meta_model,
                        self.shadow_model,
                        self.val_models_set,
                        is_discrete=is_discrete,
                        threshold="half",
                    )
                    self.args.logger.info(
                        "\tEval Loss: %.4f, AUC: %.4f, ACC: %.4f"
                        % (eval_loss, eval_auc, eval_acc)
                    )
                    if best_eval_auc is None or eval_auc > best_eval_auc:
                        best_eval_auc = eval_auc
                        model_name = "%d_model.pt" % i
                        torch.save(
                            self.meta_model.state_dict(),
                            self.args.save_folder_name / model_name,
                        )
            self.args.logger.info(f"\tEval AUC: {test_info[1] if test_info is not None else 'N/A'}")
            AUCs.append(test_info[1] if test_info is not None else 0.0)
        AUC_mean = sum(AUCs) / len(AUCs)
        self.args.logger.info(
            "Average detection AUC on %d meta classifier(val_models_set): %.4f"
            % (self.args.repeat, AUC_mean)
        )
        self.test()
