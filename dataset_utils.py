import torch
from torch.utils.data import Dataset



# Create custom datasets
class PredictorDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class SelectorDataset(Dataset):
    def __init__(self, predictions, corrects):
        self.predictions = predictions
        self.corrects = corrects

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, idx):
        return self.predictions[idx], self.corrects[idx]