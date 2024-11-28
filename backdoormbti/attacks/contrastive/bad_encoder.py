import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import copy
from torchvision import transforms
from configs.settings import BASE_DIR
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

finetune_transform_cifar10 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

finetune_transform_CLIP = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
class BadEncoder(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset

        self.trigger_input_array = np.load(BASE_DIR / args.trigger_file)
        self.target_input_array = np.load(BASE_DIR / args.reference_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x']

        self.classes = self.dataset.classes
        self.transform = train_transform
        self.bd_transform = backdoor_transform
        self.ftt_transform = finetune_transform_cifar10

    def __getitem__(self, index):
        img = self.dataset[index][0]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        img_raw = self.bd_transform(img)
        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return
    def __len__(self):
        return len(self.dataset)

