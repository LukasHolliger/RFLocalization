import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

class PhaseDiffDataSet(Dataset):

    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        self.features = features
        self.labels = labels

        self.features = torch.tensor(self.features.values, dtype=torch.float)
        self.labels = torch.tensor(self.labels.values, dtype=torch.float)

    @staticmethod
    def from_file(features_path, labels_path):
        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)

        print(labels)
        return PhaseDiffDataSet(features, labels)

    @staticmethod
    def train_validation_split(features_path: str, labels_path: str, validation_split: float):
        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=validation_split,
                                                            random_state=42)

        return PhaseDiffDataSet(X_train, y_train), PhaseDiffDataSet(X_test, y_test)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]