import torch
import math
import numpy as np
import random
from parameters import qn, q0, n, lmb
import pandas as pd

def Calcphases(SetSize, Positions):

    phases = np.zeros((SetSize, n))
    for i in range(SetSize):
        distancetoref = math.sqrt(((Positions[i, 0] - q0[0])**2 + (Positions[i, 1] - q0[1])**2 + (Positions[i, 2] - q0[2])**2))
        for j in range(n):
            distancetoantenna = math.sqrt(((Positions[i, 0] - qn[j,0])**2 + (Positions[i, 1] - qn[j,1])**2 + (Positions[i, 2] - qn[j,2])**2))
            phases[i, j] = ((distancetoantenna - distancetoref) * 2*math.pi/lmb) % (2*math.pi)
    return phases


def GeneratePos(SetSize):

    x = np.zeros((SetSize, 3))
    x[:, 2] = 0.1
    for i in range(SetSize):
        x[i, 0] = (random.random() - 0.5) * 100
        x[i, 1] = (random.random() - 0.5) * 100
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
