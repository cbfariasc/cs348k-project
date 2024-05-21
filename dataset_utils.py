import torch
from torch.utils.data import Dataset

class PredictorDataset(Dataset):
    def __init__(self, hidden_outputs, final_predictions):
        self.hidden_outputs = hidden_outputs
        self.final_predictions = final_predictions

    def __len__(self):
        return len(self.hidden_outputs)

    def __getitem__(self, idx):
        hidden_output = self.hidden_outputs[idx]
        final_prediction = self.final_predictions[idx]
        return hidden_output, final_prediction

class SelectorDataset(Dataset):
    def __init__(self, predictor_outputs, labels):
        self.predictor_outputs = predictor_outputs
        self.labels = labels

    def __len__(self):
        return len(self.predictor_outputs)

    def __getitem__(self, idx):
        predictor_output = self.predictor_outputs[idx]
        label = self.labels[idx]
        return predictor_output, label
