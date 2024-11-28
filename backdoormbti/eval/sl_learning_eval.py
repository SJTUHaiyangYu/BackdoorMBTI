"""
Supervised Learning Evaluation Module

This module provides a class for evaluating supervised learning models on clean and poisoned test data.
It supports different data types (image, text, audio, video) and calculates accuracy metrics.
"""
import torch

class SupervisedLearningEval:
    def __init__(self, clean_testloader, poison_testloader, args):
        """
        Initialize the supervised learning evaluator.

        Args:
            clean_testloader: DataLoader for clean testing data.
            poison_testloader: DataLoader for poisoned testing data.
            args: Configuration arguments containing model, device, data_type, etc.
        """
        self.model = args.model
        self.args = args
        self.clean_testloader = clean_testloader
        self.poison_testloader = poison_testloader
        self.device = args.device
        self.data_type = args.data_type
    def _test_model_for_image(self, testloader, calculate_ra=False):
        """
        Evaluate the model on image data.

        Args:
            testloader: DataLoader for the test data.
            calculate_ra: Flag to calculate robust accuracy if True.

        Returns:
            accuracy: The accuracy of the model on the test data.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                if not calculate_ra:
                    inputs, labels, *_ = data
                else:
                    inputs, *_, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def _test_model_for_text(self, testloader, calculate_ra=False):
        """
        Evaluate the model on text data.

        Args:
            testloader: DataLoader for the test data.
            calculate_ra: Flag to calculate robust accuracy if True.

        Returns:
            accuracy: The accuracy of the model on the test data.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                if not calculate_ra:
                    inputs, labels, *_ = data
                else:
                    inputs, *_, labels = data
                inputs = self.args.tokenizer(
                    inputs, padding=True, truncation=True, return_tensors="pt"
                )
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(**inputs)
                predicted = torch.argmax(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def _test_model_for_audio(self, testloader, calculate_ra=False):
        """
        Evaluate the model on audio data.

        Args:
            testloader: DataLoader for the test data.
            calculate_ra: Flag to calculate robust accuracy if True.

        Returns:
            accuracy: The accuracy of the model on the test data.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                if not calculate_ra:
                    inputs, labels, *_ = data
                else:
                    inputs, *_, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                predicted = outputs.argmax(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def _test_model_for_video(self, testloader, calculate_ra=False):
        """
        Evaluate the model on video data.

        Args:
            testloader: DataLoader for the test data.
            calculate_ra: Flag to calculate robust accuracy if True.

        Returns:
            accuracy: The accuracy of the model on the test data.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                if not calculate_ra:
                    inputs, labels, *_ = data
                else:
                    inputs, *_, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total
    def eval_model(self):
        """
        Evaluate the model on both clean and poisoned test data.

        Returns:
            tuple: A tuple containing clean accuracy, poisoned accuracy, and robust accuracy.
        """
        self.model.eval()
        with torch.no_grad():
            if self.data_type == "image":
                acc = self._test_model_for_image(self.clean_testloader)
                asr = self._test_model_for_image(self.poison_testloader)
                ra = self._test_model_for_image(self.poison_testloader, calculate_ra=True)
            elif self.data_type == "text":
                acc = self._test_model_for_text(self.clean_testloader)
                asr = self._test_model_for_text(self.poison_testloader)
                ra = self._test_model_for_text(self.poison_testloader, calculate_ra=True)
            elif self.data_type == "audio":
                acc = self._test_model_for_audio(self.clean_testloader)
                asr = self._test_model_for_audio(self.poison_testloader)
                ra = self._test_model_for_audio(self.poison_testloader, calculate_ra=True)
            elif self.data_type == "video":
                acc = self._test_model_for_video(self.clean_testloader)
                asr = self._test_model_for_video(self.poison_testloader)
                ra = self._test_model_for_video(self.poison_testloader, calculate_ra=True)
            return acc, asr, ra