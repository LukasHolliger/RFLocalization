import torch
import math
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from data import PhaseDiffDataSet
from model import PhaseDiffModel
from ObsGenerator import Calcphases
from parameters import Dmin
import matplotlib.pyplot as plt
import ObsGenerator as og
import tqdm
import random

"""
Main file:
Run this for the whole training and evaluation.
Actual main function is relatively short at the bottom, majority of this file implements the training process and then runs it.
"""


def TrainLocalization(model: PhaseDiffModel, num_epochs: int, alpha: int, sphericalcoord: bool):

    ##############################
    ### Main training pipeline ###
    ##############################

    #The triple hashtag (###) denotes lines where variables may be customized


    ###define validation_split (default 0.05), how much noise (sigma, STANDARD DEVIATION) to add to priors, how much noise (sigma2obs, VARIANCE in degrees^2) to add to observations and whether to use spherical coordinates###
    train_dataset, val_dataset, noiseaverage = PhaseDiffDataSet.train_validation_split('GeneratedPhases.csv', 'GeneratedPositions.csv', validation_split=0.05, addnoise=True, sigma=1, sigma2obs=114.6, usesphericalcoordinates=sphericalcoord)


    baselinediff = torch.square(torch.index_select(val_dataset.labels, 1, torch.tensor([0, 1, 2])) - torch.index_select(val_dataset.features, 1, torch.tensor([0, 1, 2])))
    baseline = torch.sum(baselinediff, 1, keepdim=True)/3
    baseline = torch.sum(baseline)/len(baseline)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    criterion = torch.nn.MSELoss()

    ###Define learning rate of optimizer here (this is usually the tough one to set right)###
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000072)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    outputhist = np.zeros([len(val_dataset), 3])
    labelhist = np.zeros([len(val_dataset), 3])
    for epoch in range(num_epochs):

        running_loss = 0.0
        model.train()

        # training loss
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            inputphases = inputs[0, 3:403]
            inputphases = inputphases[None, :]

            # changing spherical coordinates to cartesian for loss calculation
            if sphericalcoord:
                outputscart = torch.zeros(outputs.shape)
                labelscart = torch.zeros(labels.shape)
                outputscart[0][0] = outputs[0][0] * math.cos(outputs[0][2]) * math.sin(outputs[0][1])
                outputscart[0][1] = outputs[0][0] * math.sin(outputs[0][2]) * math.sin(outputs[0][1])
                outputscart[0][2] = outputs[0][0] * math.cos(outputs[0][1])
                labelscart[0][0] = labels[0][0] * math.cos(labels[0][2]) * math.sin(labels[0][1])
                labelscart[0][1] = labels[0][0] * math.sin(labels[0][2]) * math.sin(labels[0][1])
                labelscart[0][2] = labels[0][0] * math.cos(labels[0][1])
                labels = labelscart
                outputs = outputscart
            outputphases = torch.tensor(Calcphases(1, outputs))

            #loss function
            angleloss = criterion(inputphases, outputphases)
            loss = alpha * criterion(outputs, labels) + (1 - alpha) * angleloss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #if (i+1) % 100 == 0:
            #    print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_dataloader)}, Loss: {running_loss/(i+1):.4f}")

        # validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_losspure = 0.0
            for i, data in enumerate(val_dataloader):
                inputs, labels = data
                losshist = np.zeros(len(val_dataloader))
                outputs = model(inputs)
                val_inputphases = inputs[0, 3:403]
                val_inputphases = val_inputphases[None, :]
                restimate = torch.sqrt(outputs[0][0]**2 + outputs[0][1]**2 + outputs[0][2]**2).item()

                #changing spherical coordinates to cartesian for loss calculation
                if sphericalcoord:
                    outputscart = torch.zeros(outputs.shape)
                    labelscart = torch.zeros(labels.shape)
                    outputscart[0][0] = outputs[0][0] * math.cos(outputs[0][2]) * math.sin(outputs[0][1])
                    outputscart[0][1] = outputs[0][0] * math.sin(outputs[0][2]) * math.sin(outputs[0][1])
                    outputscart[0][2] = outputs[0][0] * math.cos(outputs[0][1])
                    labelscart[0][0] = labels[0][0] * math.cos(labels[0][2]) * math.sin(labels[0][1])
                    labelscart[0][1] = labels[0][0] * math.sin(labels[0][2]) * math.sin(labels[0][1])
                    labelscart[0][2] = labels[0][0] * math.cos(labels[0][1])
                    labels = labelscart
                    restimate = outputs[0][0].item()
                    outputs = outputscart
                val_outputphases = torch.tensor(Calcphases(1, outputs))
                val_angleloss = criterion(val_inputphases, val_outputphases)


                val_loss += alpha * criterion(outputs, labels).item() + (1 - alpha) * val_angleloss.item()
                val_losspure += criterion(outputs, labels).item()

                losshist[i] = criterion(outputs, labels).item()


                #Setting aside 40 points from last epoch to visualize results
                if epoch == num_epochs-1 and i < 40:
                    outputhist[i][0] = outputs[0][0]
                    outputhist[i][1] = outputs[0][1]
                    outputhist[i][2] = outputs[0][2]
                    labelhist[i][0] = labels[0][0]
                    labelhist[i][1] = labels[0][1]
                    labelhist[i][2] = labels[0][2]
                    if i == 39:
                        diff = outputhist[:39] - labelhist[:39]
                        #print(diff)
                        sampleMSE = np.zeros(len(diff))
                        for i in range(len(sampleMSE)):
                            sampleMSE[i] = 1/3 * (diff[i][0]**2 + diff[i][1]**2 + diff[i][2]**2)
                            print(f"SampleMSE:{sampleMSE[i]}")

                        print(f"Average MSE:{np.sum(sampleMSE)/len(sampleMSE)}\nVal_loss:{val_loss}")

            val_loss /= len(val_dataloader)
            val_losspure /= len(val_dataloader)
            rmse_loss = math.sqrt(val_loss)
            var = 1/np.sqrt(len(val_dataloader))*np.sqrt(np.sum(np.square(losshist - val_loss)))

        #Plot last epoch
        if epoch == num_epochs-1:
            plt.scatter(outputhist[:39, 0], outputhist[:39, 1], facecolors='none',
                        edgecolors='b')
            plt.scatter(labelhist[:39, 0], labelhist[:39, 1], facecolors='none', edgecolors='r', alpha=0.5)
            plt.title('Original positions (red) vs Estimates (blue)')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}, Validation Loss: {10 * np.log10(val_loss):.4f}[dB], Noise Loss: {10 * np.log10(alpha * noiseaverage):.4f}[dB], Second order: {10 * np.log10(var + val_loss) - 10 * np.log10(val_loss):.4f}")
    print(baseline, noiseaverage)

    #Pure loss disregards custom alpha values and sets it to 1. Has no influence on training process though.
    print(f"Pure validation loss: {10 * np.log10(val_losspure):.4f}, Pure Noise loss: {10 * np.log10(noiseaverage):.4f}")
    print(f"Amount of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return


random.seed(10)

if False:                                       #Set to True to generate new Observations
    og.GenerateObservations(25000, True)        #Set to True to write new observations into CSV files

model = PhaseDiffModel()

#Define number of epochs for training process here
TrainLocalization(model, num_epochs=30, alpha=1, sphericalcoord=False)
torch.save(model.state_dict(), 'model.pt')




