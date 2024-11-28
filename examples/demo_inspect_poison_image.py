import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

transform = transforms.Compose([transforms.ToTensor()])
dataset = CIFAR10(root="./data/cifar10", download=True, train=True, transform=transform)
poisonset = torch.load("./image_badnet_poison_train_set.pt")
# print(type(poisonset))
print(poisonset[0][0].shape)
# print(poisonset[0])

# convert tensor to image
transform = transforms.ToPILImage()

# save image
img = transform(poisonset[0][0])
img.save("poison_image.png")
img = transform(dataset[0][0])
img.save("benign_image.png")
