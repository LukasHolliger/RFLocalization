import math
import numpy as np
import random
from parameters import qn, q0, n, lmb, Dmax, Dmin, Q_mod, tau
from ObsGenerator import GeneratePos, Calcphases
import pandas as pd

"""
Used by Evaluate Trajectory
Set up to be a constant velocity model. Edit Velocities[0][x] lines to change possible starting values. "(random.random() - 0.5)*0.2" for example is anywhere between -0.1 and 0.1
"""

def GenerateTraj(Length):

    x_0 = GeneratePos(1)

    Positions = np.zeros((Length, 3))
    Positions[0] = x_0
    Velocities = np.zeros((Length, 3))
    Velocities[0][0] = (random.random() - 0.5)*0.2
    Velocities[0][1] = (random.random() - 0.5)*0.2
    Velocities[0][2] = (random.random() - 0.5)*0

    for i in range(Length-1):
        #eq = np.random.multivariate_normal(np.zeros(6), Q_mod, 1)
        Positions[i+1] = Positions[i] + tau*Velocities[i] #+ eq[0,0:3]
        Velocities[i + 1] = Velocities[i]
        #Velocities[i+1] = Velocities[i] + q**2 * np.random.normal(0, tau, size=Velocities[i].shape) + eq[0,3:6]

    Phases = Calcphases(Length, Positions)

    PositionsDF = pd.DataFrame(Positions)
    PhasesDF = pd.DataFrame(Phases)
    VelocitiesDF = pd.DataFrame(Velocities)
    PositionsDF.to_csv("TrajPositions.csv", index=False)
    PhasesDF.to_csv("TrajPhases.csv", index=False)
    VelocitiesDF.to_csv("TrajVelocities.csv", index=False)

    return Positions, Phases, Velocities



#Pos, Phas, V = GenerateTraj(50)