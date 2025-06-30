import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

os.chdir(sys.path[0])
sys.path.append("../../")
os.getcwd()

from utils.data import load_dataset


class PGD:
    def __init__(
        self,
        model,
        eps=12 / 255,
        alpha=2 / 255,
        steps=10,
        device="cpu",
        random_start=True,
        normalize=None,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.random_start = random_start
        self.normalize = normalize

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.normalize(adv_images))

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images, delta

    def __call__(self, images, labels, *args, **kwargs):
        return self.forward(images, labels, *args, **kwargs)


def gen_adv_data(
    dataset_name, dataset_path, clean_model_path, save_folder, device, args
):

    args.dataset = dataset_name
    args.dataset_path = os.path.join(dataset_path, dataset_name)
    train_dataset_without_transform = load_dataset(args, train=True)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(clean_model_path):
        os.makedirs(clean_model_path)
    if "clean.pth" not in os.listdir(clean_model_path):
        match dataset_name:
            case "cifar10":
                model = torch.hub.load(
                    "chenyaofo/pytorch-cifar-models",
                    "cifar10_resnet20",
                    pretrained=True,
                )
            case _:
                raise ValueError(
                    "please train the clean model, name it as 'clean.pth', and place it in resources/{}".format(
                        dataset_name
                    )
                )
    else:
        from torchvision.models import resnet18

        model = resnet18(num_classes=args.num_classes)
        model.load_state_dict(torch.load(os.path.join(clean_model_path, "clean.pth")))
    model.eval()
    model.to(device)

    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    atk = PGD(
        model, eps=12 / 255, alpha=2 / 255, steps=7, device=device, normalize=normalize
    )
    loader = DataLoader(dataset=train_dataset_without_transform, batch_size=256)
    acc = 0.0
    for i, (images, labels) in enumerate(tqdm(loader)):
        adv_imgs, _ = atk(images, labels)
        if i == 0:
            adv_data = adv_imgs
        else:
            adv_data = torch.cat((adv_data, adv_imgs), dim=0)
        with torch.no_grad():
            out = model(normalize(adv_imgs))
            acc += (out.argmax(dim=1) == labels.to(device)).sum().item()
    acc /= len(train_dataset_without_transform)
    print("acc on adv imgs: ", acc)
    adv_data = (adv_data * 255).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    np.save(os.path.join(save_folder, "train.npy"), adv_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset into images")
    parser.add_argument(
        "--dataset_name", type=str, default="cifar10", help="dataset name"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="../../data", help="dataset path"
    )
    parser.add_argument("--device", type=str, default="cuda", help="clean model path")
    args = parser.parse_args()
    args.attack_train_replace_imgs_path = "../../data/labelconsistent/{dataset}".format(
        dataset=args.dataset_name
    )
    args.clean_model_path = "./{dataset}".format(dataset=args.dataset_name)
    gen_adv_data(
        args.dataset_name,
        args.dataset_path,
        args.clean_model_path,
        args.attack_train_replace_imgs_path,
        args.device,
    )
