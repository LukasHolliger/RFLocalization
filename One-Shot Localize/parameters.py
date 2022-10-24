import torch
import math
import numpy as np

#########################
### Design Parameters ###
#########################
n = 100 # amount of antenna elements
m = 6  # dimension of our state space

m1x_0 = torch.ones(m, 1)
m1x_0 = torch.transpose(torch.tensor([[2.5, -9.1, 1.5, 0.01, 0.97, 0]], dtype=torch.float), 0, 1)
P_0 = m1x_0 + torch.transpose(torch.tensor([np.random.normal(0, [0.5**2, 0.5**2, 0.01**2, 0.01**2/10, 0.97**2/10, 0])], dtype=torch.float), 0, 1)
m2x_0 = torch.zeros(m, m)
#P_0 = P_0[None, :]

T = 20  # Amount of timesteps
T_test = 20
lmb = 0.010714285714286  # Wavelength
Dmin = lmb/(2*math.pi)


########################
### Model Parameters ###
########################
tau = 1  # temporal step

A = torch.eye(6)
A[0:3, 3:6] = torch.mul(torch.eye(3), tau)  # State Transition Matrix


B = torch.zeros(3, m)
B[0:3, 0:3] = torch.eye(3)

# Noise Parameters
Qx = torch.tensor([[0.03**2,  0, 0], [0, 0.03**2, 0], [0,  0, 0]])
sigma_r_mod = 0.349065850398866  # noise standard deviation in radians (corresponds to 20°)

# Noise Matrices
Q_aux = torch.block_diag(torch.mul(Qx, pow(tau, 2)/2), torch.mul(Qx, pow(tau, 2)/2))  # auxiliary matrix to define transition covariance matrix
Q_mod = torch.block_diag(torch.mul(Qx, pow(tau, 3)/3), torch.mul(Qx, tau))
Q_mod[3:6, 0:3] = Q_aux[0:3, 0:3]
Q_mod[0:3, 3:6] = Q_aux[0:3, 0:3]  # Finalized transition covariance matrix

R_mod = sigma_r_mod * torch.eye(n)  # Observation covariance matrix

#########################
### Antenna positions ###
#########################

qn = torch.zeros(n, 3)  # Antenna positions
dqn = 0.005357142857143  # distance between individual array elements
sqrtn = math.sqrt(n)
for i in range(n):
    qn[i][2] = dqn * (i % sqrtn)
    qn[i][1] = dqn * math.trunc(i / sqrtn)

Dmax = 2*(19*dqn*math.sqrt(2)/2)**2/lmb  # Maximum Near-Field distance ("sqrtn-1" changed to 19 to generate area of points similar to other paper
q0 = torch.tensor([0, (math.sqrt(n)-1)/2*dqn, (math.sqrt(n)-1)/2*dqn])  # Reference position