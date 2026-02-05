from torch import tensor
from torch.utils.data import Dataset
from torch import load
from torch.nn import functional as F


class dataset(Dataset):
    def __init__(self, subjects):
        self.samples = []
        self.labels = []
        counter = 1
        padding_size = 0
        for subject in subjects:
            data = load(subject)
            if data.shape[2] > padding_size:
                padding_size = 0
            samples = [data[i] for i in range(data.shape[0])]
            labels = [1] * len(data.shape[0])
            counter += 1
            self.samples += samples
            self.labels += labels
        for sample in samples:
            F.pad(sample, (0, padding_size - len(sample)), value=-1)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return (self.samples[index], self.labels[index])