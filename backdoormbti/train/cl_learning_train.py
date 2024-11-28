'''
This file provides a structured approach to training contrastive learning models for different data types.
'''
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
from datetime import datetime
import copy
from utils.optim import get_lr_scheduler, get_optimizer
class ContrastiveLearningTrain:
    def __init__(self, trainloader, args):
        """
        Initialize the contrastive learning trainer with necessary components.

        Args:
            trainloader: DataLoader for training data.
            args: Configuration arguments containing model, optimizer, scheduler, device, epochs, etc.
        """
        self.model = args.model
        self.clean_model = copy.deepcopy(args.model)
        self.args = args
        self.args.knn_t = 0.5
        self.trainloader = trainloader
        self.optimizer = get_optimizer(
            args.client_optimizer,
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.criterion = torch.nn.CrossEntropyLoss()  
        if args.lr_scheduler is not None:
            self.lr_scheduler = get_lr_scheduler(
                args.lr_scheduler, self.optimizer, args
            )
        else:
            self.lr_scheduler = None
        self.args.lambda1 = 1.0
        self.args.lambda2 = 1.0
        self.device = args.device
        self.epochs = args.epochs
        self.data_type = args.data_type
    def _train_model_for_image_image(self):
        """
        Train the model on image & image data.
        """
        self.model.train()
        for epoch in range(self.epochs):
            start_time = datetime.now()
            running_loss = 0.0
            for i, data in enumerate(
                tqdm(
                    self.trainloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="batch",
                )
            ):
                image_1, image_2 = data
                image_1, image_2 = image_1.cuda(non_blocking=True), image_2.cuda(non_blocking=True)
                feature_1, out_1 = self.model(image_1)
                feature_2, out_2 = self.model(image_2)
                # [2*B, D]
                out = torch.cat([out_1, out_2], dim=0)
                # [2*B, 2*B]
                sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.args.knn_t)
                mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.args.batch_size, device=sim_matrix.device)).bool()
                # [2*B, 2*B-1]
                sim_matrix = sim_matrix.masked_select(mask).view(2 * self.args.batch_size, -1)

                # compute loss
                pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.args.knn_t)
                # [2*B]
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
                # loss = net(im_1, im_2, args)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
            end_time = datetime.now()
            time_taken = (end_time - start_time).total_seconds()
            lr = self.optimizer.param_groups[0]["lr"]

            log_message = f"Train Epoch: {epoch+1:02d}\tLoss: {avg_loss:.6f}\tTime taken: {time_taken:.3f}s\tLearning Rate: {lr:.8f}"
            print(log_message)
            #self.args.logger.info(log_message)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    def _train_bkd_model_for_bad_encoder(self, epoch):

        self.model.train()

        for module in self.model.modules():
        # print(module)
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

        self.clean_model.eval()

        total_loss, total_num, train_bar = 0.0, 0, tqdm(self.trainloader)
        total_loss_0, total_loss_1, total_loss_2 = 0.0, 0.0, 0.0
        for img_clean, img_backdoor_list, reference_list,reference_aug_list in train_bar:
            img_clean = img_clean.cuda(non_blocking=True)
            reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
            for reference in reference_list:
                reference_cuda_list.append(reference.cuda(non_blocking=True))
            for reference_aug in reference_aug_list:
                reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
            for img_backdoor in img_backdoor_list:
                img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))

            clean_feature_reference_list = []

            with torch.no_grad():
                clean_feature_raw = self.clean_model(img_clean)
                clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
                for img_reference in reference_cuda_list:
                    clean_feature_reference = self.clean_model(img_reference)
                    clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                    clean_feature_reference_list.append(clean_feature_reference)

            feature_raw = self.model(img_clean)
            feature_raw = F.normalize(feature_raw, dim=-1)

            feature_backdoor_list = []
            for img_backdoor in img_backdoor_cuda_list:
                feature_backdoor = self.model(img_backdoor)
                feature_backdoor = F.normalize(feature_backdoor, dim=-1)
                feature_backdoor_list.append(feature_backdoor)

            feature_reference_list = []
            for img_reference in reference_cuda_list:
                feature_reference = self.model(img_reference)
                feature_reference = F.normalize(feature_reference, dim=-1)
                feature_reference_list.append(feature_reference)

            feature_reference_aug_list = []
            for img_reference_aug in reference_aug_cuda_list:
                feature_reference_aug = self.model(img_reference_aug)
                feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
                feature_reference_aug_list.append(feature_reference_aug)

            loss_0_list, loss_1_list = [], []
            for i in range(len(feature_reference_list)):
                loss_0_list.append(- torch.sum(feature_backdoor_list[i] * feature_reference_list[i], dim=-1).mean())
                loss_1_list.append(- torch.sum(feature_reference_aug_list[i] * clean_feature_reference_list[i], dim=-1).mean())
            loss_2 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()

            loss_0 = sum(loss_0_list)/len(loss_0_list)
            loss_1 = sum(loss_1_list)/len(loss_1_list)

            loss = loss_0 + self.args.lambda1 * loss_1 + self.args.lambda2 * loss_2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_num += self.trainloader.batch_size
            total_loss += loss.item() * self.trainloader.batch_size
            total_loss_0 += loss_0.item() * self.trainloader.batch_size
            total_loss_1 += loss_1.item() * self.trainloader.batch_size
            total_loss_2 += loss_2.item() * self.trainloader.batch_size
            train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(epoch, self.epochs, self.optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num))

        return total_loss / total_num
    def train_model(self):
        """
        Dispatch the training process based on the data type.
        """
        
        if self.data_type == "contrastive_learning": 
            for epoch in range(self.epochs):
                self._train_bkd_model_for_bad_encoder(epoch)
        elif self.data_type == "image_text":
            self._train_model_for_image_text()
        else:
            self._train_bkd_model_for_bad_encoder()
        