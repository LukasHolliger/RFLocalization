import torch
import torch.nn as nn
from parameters import n

NUM_FEATURES = n

class PhaseDiffModel(nn.Module):
    def __init__(self, num_features=NUM_FEATURES):
        super(PhaseDiffModel, self).__init__()
        #self.conv1 = torch.nn.Conv1d(1, 1, kernel_size=2, stride=1)
        #self.pool = torch.nn.AdaptiveAvgPool1d(32)
        self.fc = nn.Linear(NUM_FEATURES, 32)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 3)
        #self.fc3 = nn.Linear(16, 3)
        #self.fc4 = nn.Linear(8, 3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)


        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.fc(x)
        #out = self.conv1(x)
        #out = self.pool(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        #out = self.bn1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        #out = self.relu(out)
        #out = self.bn2(out)
        #out = self.dropout(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        #out = self.fc4(out)


        return out

