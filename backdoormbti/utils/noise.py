'''
a file for add noise to normal data
'''
from networkx import random_cograph
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import random
from torch.utils.data import Dataset
import torch
augumenter_list = [
    "ocr",
    "keyboard",
    "random_char_insert",
    "random_char_substitute",
    "random_char_swap",
    "spelling",
    "word_embedding_wordnet",
    "contextual_word_insert",
    "contextual_word_substitute",
    "wordnet_word_substitute",
]
from nlpaug.util import Action

def get_noise_processing(dataset, args, train):
    
    if hasattr(args, 'add_noise') and args.add_noise and train:
        match args.noise_type:
            case "gaussian_noise":
                ##TODO： 异常处理
                noisy_dataset = GaussianNoiseDataset(dataset=dataset, 
                                                     noise_ratio=args.noise_ratio,
                                                     noise_mean=args.noise_mean,
                                                     noise_std=args.noise_std)
            case "text_noise":
                if args.dataset == "sst2" :
                    def noise_formator(data):
                        text, label = data
                        if torch.rand(1) < args.noise_ratio:
                            text = add_text_noise(text)
                        return text, label
                
                else:
                    def noise_formator(data):
                        label, text = data
                        if torch.rand(1) < args.noise_ratio:
                            text = add_text_noise(text)
                        return text, label - 1
                noisy_dataset = dataset.map(noise_formator)
        return noisy_dataset
    else:
        return dataset
def add_text_noise(text):
    return text_augment(text, random.choice(augumenter_list))

class GaussianNoiseDataset(Dataset):
    def __init__(self, dataset, noise_ratio=0.1, noise_mean=0.0, noise_std=0.1):
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.noise_mean = noise_mean
        self.noise_std = noise_std


        self.num_samples = len(self.dataset)
        self.noisy_indices = random.sample(range(self.num_samples), int(self.num_samples * self.noise_ratio))

    def add_gaussian_noise(self, image):

        noise = torch.randn_like(image) * self.noise_std +self.noise_mean
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0.0, 1.0)  # 保证像素值范围为 [0, 1]

    def __getitem__(self, index):
        img, target = self.dataset[index]  # 获取原始数据
        if index in self.noisy_indices:
            img = self.add_gaussian_noise(img)
        return img, target

    def __len__(self):
        return len(self.dataset)

def text_augment(text, augmenter):
    if augmenter == "ocr":
        aug = nac.OcrAug()
        augmented_text = aug.augment(text)
    elif augmenter == "keyboard":
        aug = nac.KeyboardAug()
        augmented_text = aug.augment(text)
    elif augmenter == "random_char_insert":
        aug = nac.RandomCharAug(action="insert")
        augmented_text = aug.augment(text)
    elif augmenter == "random_char_substitute":
        aug = nac.RandomCharAug(action="substitute")
        augmented_text = aug.augment(text)
    elif augmenter == "random_char_swap":
        aug = nac.RandomCharAug(action="swap")
        augmented_text = aug.augment(text)
    elif augmenter == "spelling":
        aug = naw.SpellingAug()
        augmented_text = aug.augment(text)
    elif augmenter == "word_embedding_wordnet":
        aug = naw.SynonymAug(aug_src="wordnet")
        augmented_text = aug.augment(text)
    elif augmenter == "contextual_word_insert":
        aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", action="insert")
        augmented_text = aug.augment(text)
    elif augmenter == "contextual_word_substitute":
        aug = naw.ContextualWordEmbsAug(
            model_path="bert-base-uncased", action="substitute"
        )
        augmented_text = aug.augment(text)
    elif augmenter == "wordnet_word_substitute":
        aug = naw.SynonymAug(aug_src="wordnet")
        augmented_text = aug.augment(text)
    return augmented_text
