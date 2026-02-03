from torch import tensor
from torch.utils.data import Dataset
from torch import load


class dataset(Dataset):
    def __init__(self, subjects):
        self.samples = []
        self.labels = []
        counter = 1
        for subject in subjects:
            data = load(subject)
            samples = [data[i] for i in range(data.shape[0])]
            labels = [1] * len(data.shape[0])
            counter += 1
            self.samples += samples
            self.labels += labels
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return (self.samples[index], self.labels[index])