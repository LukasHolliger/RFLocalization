import torch
import math
import numpy as np
import random
from parameters import qn, q0, n, lmb, Dmax, Dmin
import pandas as pd

"""
ObsGenerator contains important functions such as Calcphases (observation function) and GeneratePos to create a dataset.
GeneratePos can be changed to define different ranges for the dataset, more details down below.
"""
# Calcphases corresponds to the observation function H that translates original positions to phase readings at the n antenna array elements
def Calcphases(SetSize, Positions):

    phases = np.zeros((SetSize, n))
    if SetSize > 1:
         for i in range(SetSize):
            distancetoref = math.sqrt(((Positions[i, 0] - q0[0])**2 + (Positions[i, 1] - q0[1])**2 + (Positions[i, 2] - q0[2])**2))
            for j in range(n):
                distancetoantenna = math.sqrt(((Positions[i, 0] - qn[j,0])**2 + (Positions[i, 1] - qn[j,1])**2 + (Positions[i, 2] - qn[j,2])**2))
                phases[i, j] = ((distancetoantenna - distancetoref) * 2*math.pi/lmb) % (2*math.pi)
    if SetSize == 1:
        distancetoref = math.sqrt(((Positions[0] - q0[0]) ** 2 + (Positions[1] - q0[1]) ** 2 + (Positions[2] - q0[2]) ** 2))
        phases = np.zeros(n)
        for j in range(n):
            distancetoantenna = math.sqrt(((Positions[0] - qn[j, 0]) ** 2 + (Positions[1] - qn[j,1]) ** 2 + (Positions[2] - qn[j, 2]) ** 2))
            phases[j] = ((distancetoantenna - distancetoref) * 2 * math.pi / lmb) % (2 * math.pi)
    return phases


#diffphases is deprecated
def diffphases(phases):
    phasevariance = (2 * math.pi / 360) * 0
    phasenoise = np.random.normal(0, np.sqrt(phasevariance), size=phases.shape)
    noisedphases = phases.add(phasenoise) % 2 * math.pi
    apd = (math.sqrt(n)*(math.sqrt(n)-1))
    phasedifferences = np.zeros((len(phases)/n, 2*apd))
    for i in range(len(phases)/n):
        for j in range(n):
            if j % 20 != 19:
                phasedifferences[i, j] = noisedphases[i, j] - noisedphases[i, j+1]
                if phasedifferences[i, j] > math.pi:
                    phasedifferences[i, j] = noisedphases[i, j + 1] + 2 * math.pi - noisedphases[i, j]
                if phasedifferences[i, j] < -math.pi:
                    phasedifferences[i, j] = -(noisedphases[i, j] + 2 * math.pi - noisedphases[i, j + 1])
            if j < 380:
                phasedifferences[i, j + apd] = noisedphases[i, j] - noisedphases[i, j + 20]
                if phasedifferences[i, j + apd] > math.pi:
                    phasedifferences[i, j + apd] = noisedphases[i, j + 20] + 2 * math.pi - noisedphases[i, j]
                if phasedifferences[i, j + apd] < -math.pi:
                    phasedifferences[i, j + apd] = -(noisedphases[i, j] + 2 * math.pi - noisedphases[i, j + 20])

    return phasedifferences


def GeneratePos(SetSize):

    x = np.zeros((SetSize, 3))

    #z coordinate defined as 0.1 like in large antenna array paper, small noise added to it so it's not completely static
    x[:, 2] = 0.1
    x[:, 2] += np.random.normal(0, 0.01, size=x[:, 2].shape)


    for i in range(SetSize):

        #x coordinate anywhere between 0 and (3 * Dmax - Dmin)
        x[i, 0] = random.random() * (3 * Dmax - Dmin)

        #add Dmin as minimum distance
        if x[i, 0] < 0:
            x[i, 0] = x[i, 0] - Dmin
        else:
            x[i, 0] = x[i, 0] + Dmin

        #y coordinate anywhere between 0 and (3 * Dmax - Dmin)
        x[i, 1] = random.random() * (3 * Dmax - Dmin)

        # add Dmin as minimum distance
        if x[i, 1] < 0:
            x[i, 1] = x[i, 1] - Dmin
        else:
            x[i, 1] = x[i, 1] + Dmin
    return x


def GenerateObservations(SetSize, tocsv=False):
    Positions = GeneratePos(SetSize)
    Phases = Calcphases(SetSize, Positions)
    if tocsv:
        PositionsDF = pd.DataFrame(Positions)
        PhasesDF = pd.DataFrame(Phases)
        PositionsDF.to_csv("GeneratedPositions.csv", index=False)
        PhasesDF.to_csv("GeneratedPhases.csv", index=False)


    return Positions, Phases

def GenerateDiffPhasescsv():
    phases = pd.read_csv("GeneratedPhases.csv")
    PhaseDiffDF = pd.DataFrame(diffphases(phases))
    PhaseDiffDF.to_csv("PhaseDifferences.csv")

#GenerateObservations(25000, True)