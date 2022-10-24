import math
import torch
import torch.nn as nn
from parameters import n

NUM_FEATURES = n

class PhaseDiffModel(nn.Module):
    def __init__(self, num_features=NUM_FEATURES):
        super(PhaseDiffModel, self).__init__()
        #self.pool = torch.nn.AvgPool2d(4, 4)
        self.fc = nn.Linear(NUM_FEATURES, 150)
        self.fc1 = nn.Linear(153, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 3)


        self.fconlyobs = nn.Linear(60, 3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)


        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        #separate prior from features
        estimate = x[0][0:3]

        out = self.fc(x[0][3:])
        out = self.relu(out)
        #out = self.dropout(out)

        #concatenate priors again
        out = torch.cat((estimate, out))
        out = self.fc1(out)
        out = self.relu(out)
        #out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        #out = self.dropout(out)

        out = self.fc3(out)
        #out = self.fconlyobs(out)
        out = out[None, :]
        #out = self.relu(out)
        #out = self.fc4(out)


        return out

