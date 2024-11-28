"""
Contrastive Learning Evaluation Module

This module provides a class for evaluating contrastive learning models on clean and poisoned test data.
It supports different data types and calculates accuracy metrics.
"""
import torch
from tqdm import tqdm
import torch.nn.functional as F
class ContrastiveLearningEval:
    def __init__(self, memory_loader, poison_testloader, args):
        """
        Initialize the supervised learning evaluator.

        Args:
            clean_testloader: DataLoader for clean testing data.
            poison_testloader: DataLoader for poisoned testing data.
            args: Configuration arguments containing model, device, data_type, etc.
        """
        self.model = args.model.f
        self.args = args
        self.args.knn_t = 0.5
        self.args.knn_k = 200
        self.testloader = poison_testloader
        self.memory_loader = memory_loader
        self.device = args.device
        self.data_type = args.data_type
    def _knn_predict(self,feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels
    def _test_model_for_image_image(self, trainloader,testloader, calculate_ra=False):
        """
        Evaluate the model on image data.

        Args:
            testloader: DataLoader for the test data.
            calculate_ra: Flag to calculate robust accuracy if True.

        Returns:
            accuracy: The accuracy of the model on the test data.
        """
        self.model.eval()
        classes = 10
        total_top1, total_num, feature_bank = 0.0, 0, []
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm(trainloader, desc='Feature extracting'):

                feature = self.model(data.cuda(non_blocking=True))

                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = torch.tensor(trainloader.dataset.targets, device=feature_bank.device)
            # loop test data to predict the label by weighted knn search
            for data, target in tqdm(testloader):
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = self.model(data)
                feature = F.normalize(feature, dim=1)

                pred_labels = self._knn_predict(feature, feature_bank, feature_labels, classes, self.args.knn_k, self.args.knn_t)
                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()

        return total_top1 / total_num 
        
    def eval_model(self):
        """
        Evaluate the model on both clean and poisoned test data.

        Returns:
            tuple: A tuple containing clean accuracy, poisoned accuracy, and robust accuracy.
        """
        self.model.eval()
        with torch.no_grad():
            if self.data_type == "contrastive_learning":
                acc = self._test_model_for_image_image(self.memory_loader, self.testloader)
                # asr = self._test_model_for_image_image(self.memory_loader, self.testloader)
                # ra = self._test_model_for_image_image(self.memory_loader, self.testloader)
            return acc
            # return acc, asr, ra