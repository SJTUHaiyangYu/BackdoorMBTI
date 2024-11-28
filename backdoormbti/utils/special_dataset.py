from torch.utils.data import DataLoader, Dataset
class BadEncoderTrainset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        x, y = data
        return x, y, 0, y