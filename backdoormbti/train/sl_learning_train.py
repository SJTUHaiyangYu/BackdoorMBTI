"""
This class provides a structured approach to training supervised learning models for different data types.
"""

from datetime import datetime

import torch
from tqdm import tqdm

from utils.optim import get_lr_scheduler, get_optimizer


def _non_blocking_device_transfer(device):
    return torch.cuda.is_available() and str(device).startswith("cuda")


class SupervisedLearningTrain:
    def __init__(self, trainloader, args):
        """
        Initialize the supervised learning trainer with necessary components.

        Args:
            trainloader: DataLoader for training data.
            args: Configuration arguments containing model, optimizer, scheduler, device, epochs, etc.
        """
        self.model = args.model
        self.args = args
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
        self.device = args.device
        self.epochs = args.epochs
        self.data_type = args.data_type
        self.non_blocking = _non_blocking_device_transfer(self.device)

    def _move_to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=self.non_blocking)
        if hasattr(value, "to"):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move_to_device(item) for key, item in value.items()}
        return value

    def _prepare_audio_inputs(self, inputs):
        inputs = self._move_to_device(inputs)
        if getattr(self.args, "audio_preprocess_on_device", False):
            inputs = self.args.pre_trans(inputs)
        return inputs

    def _train_model_for_image(self):
        """
        Train the model on image data.
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
                inputs, labels, *_ = data
                inputs = self._move_to_device(inputs)
                labels = self._move_to_device(labels)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
            end_time = datetime.now()
            time_taken = (end_time - start_time).total_seconds()
            lr = self.optimizer.param_groups[0]["lr"]

            log_message = f"Train Epoch: {epoch+1:02d}\tLoss: {avg_loss:.6f}\tTime taken: {time_taken:.3f}s\tLearning Rate: {lr:.8f}"
            self.args.logger.info(log_message)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _train_model_for_text(self):
        """
        Train the model on text data.
        """
        self.model.train()  # 切换到训练模式
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
                text, labels, *_ = data
                inputs = self.args.tokenizer(
                    text, padding=True, truncation=True, return_tensors="pt"
                )
                inputs["labels"] = labels
                inputs = self._move_to_device(inputs)
                self.optimizer.zero_grad()  # 每次迭代前清除梯度
                outputs = self.model(**inputs)  # 前向传播
                loss = outputs.loss  # 获取损失值
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
            end_time = datetime.now()
            time_taken = (end_time - start_time).total_seconds()
            lr = self.optimizer.param_groups[0]["lr"]

            log_message = f"Train Epoch: {epoch+1:02d}\tLoss: {avg_loss:.6f}\tTime taken: {time_taken:.3f}s\tLearning Rate: {lr:.8f}"
            self.args.logger.info(log_message)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _train_model_for_audio(self):
        """
        Train the model on audio data.
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

                inputs, labels, *_ = data
                inputs = self._prepare_audio_inputs(inputs)
                labels = self._move_to_device(labels)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
            end_time = datetime.now()
            time_taken = (end_time - start_time).total_seconds()
            lr = self.optimizer.param_groups[0]["lr"]

            log_message = f"Train Epoch: {epoch+1:02d}\tLoss: {avg_loss:.6f}\tTime taken: {time_taken:.3f}s\tLearning Rate: {lr:.8f}"
            self.args.logger.info(log_message)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _train_model_for_video(self):
        """
        Train the model on video data.
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

                inputs, labels, *_ = data
                inputs = self._move_to_device(inputs)
                labels = self._move_to_device(labels)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
            end_time = datetime.now()
            time_taken = (end_time - start_time).total_seconds()
            lr = self.optimizer.param_groups[0]["lr"]

            log_message = f"Train Epoch: {epoch+1:02d}\tLoss: {avg_loss:.6f}\tTime taken: {time_taken:.3f}s\tLearning Rate: {lr:.8f}"
            self.args.logger.info(log_message)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def train_model(self):
        """
        Dispatch the training process based on the data type.
        """
        if self.data_type == "image":
            self._train_model_for_image()
        elif self.data_type == "text":
            self._train_model_for_text()
        elif self.data_type == "audio":
            self._train_model_for_audio()
        elif self.data_type == "video":
            self._train_model_for_video()
