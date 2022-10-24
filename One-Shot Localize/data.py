import torch
import math
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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

        return PhaseDiffDataSet(features, labels)

    @staticmethod
    def train_validation_split(features_path: str, labels_path: str, validation_split: float, addnoise: bool, sigma: float, sigma2obs: float, usesphericalcoordinates: False):
        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)
        np.random.seed(10)
        if addnoise == True:

            #add noise to observations
            phasevariance = (2*math.pi/360)**2*sigma2obs    #Final value in this line represents variance of observation noise in degrees^2
            phasenoise = np.random.normal(0, np.sqrt(phasevariance), size=features.shape)
            noisefeatures = features.add(phasenoise) % 2*math.pi

            #dev = (labels**2).sum()/25000
            #dev = dev**(1/2)
            #x_dev = dev[0]*sigma
            #y_dev = dev[1]*sigma
            #z_dev = dev[2]*sigma
            #print(x_dev, y_dev, z_dev)

            #generate noise for priors
            noise_x = np.random.normal(0, sigma, size=labels['0'].shape)
            noise_y = np.random.normal(0, sigma, size=labels['0'].shape)
            noise_z = np.random.normal(0, sigma*0.01, size=labels['0'].shape)
            noise = np.vstack((noise_x, noise_y, noise_z)).transpose()

            noiseaverage = np.sum(np.abs(noise), 0, keepdims=True)
            noiseaverage = (np.sum(noiseaverage)/3)/len(noise)

            print(f"Prior Noise level(dB):{10*np.log10(sigma)}\n")

            #add noise to priors and concatenate to features
            labfeatures = pd.concat([labels.add(noise), noisefeatures], join='outer', axis=1)

            #change to cartesian to spherical coordinates if set to True
            if usesphericalcoordinates:
                noisedlabels = labels.add(noise)
                sphericlabels = np.zeros(noisedlabels.shape)
                for i in range(len(sphericlabels)):
                    sphericlabels[i, 0] = np.sqrt(
                        np.square(noisedlabels.at[i, '0']) + np.square(noisedlabels.at[i, '1']) + np.square(noisedlabels.at[i, '2']))
                    sphericlabels[i, 1] = np.arccos(labels.at[i, '2'] / sphericlabels[i, 0])
                    if noisedlabels.at[i, '0'] > 0:
                        sphericlabels[i, 2] = np.arctan(noisedlabels.at[i, '1'] / noisedlabels.at[i, '0'])
                    if noisedlabels.at[i, '0'] < 0 and noisedlabels.at[i, '1'] > 0:
                        sphericlabels[i, 2] = np.arctan(noisedlabels.at[i, '1'] / noisedlabels.at[i, '0']) + math.pi
                    if noisedlabels.at[i, '0'] < 0 and noisedlabels.at[i, '1'] < 0:
                        sphericlabels[i, 2] = np.arctan(noisedlabels.at[i, '1'] / noisedlabels.at[i, '0']) - math.pi
                    if noisedlabels.at[i, '0'] == 0 and noisedlabels.at[i, '1'] > 0:
                        sphericlabels[i, 2] = math.pi / 2
                    if noisedlabels.at[i, '0'] == 0 and noisedlabels.at[i, '1'] < 0:
                        sphericlabels[i, 2] = -math.pi / 2
                sphericlabelsDF = pd.DataFrame(sphericlabels)
                labfeatures = pd.concat([sphericlabelsDF, noisefeatures], join='outer', axis=1)
                labels = sphericlabelsDF

        else:
            labfeatures = pd.concat([labels, features], join='outer', axis=1)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(labfeatures, labels, test_size=validation_split,
                                                            random_state=42)

        return PhaseDiffDataSet(X_train, y_train), PhaseDiffDataSet(X_test, y_test), noiseaverage

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]