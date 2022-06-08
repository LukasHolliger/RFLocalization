import torch
import math

#########################
### Design Parameters ###
#########################
n = 400
m = 3  # dimension of our state space

m1x_0 = torch.ones(m, 1)
m2x_0 = torch.zeros(m, m)

q0 = torch.tensor([0, 0, 1])  # Reference position
T = 10  # Amount of timesteps
T_test = 10
lmb = 0.010714285714286  # Wavelength

########################
### Model Parameters ###
########################
tau = 1  # temporal step

A = torch.eye(6)
A[0:3, 3:6] = torch.mul(torch.eye(3), tau)  # State Transition Matrix

B = torch.zeros(3, m)
B[0:3, 0:3] = torch.eye(3)

# Noise Parameters
Qx = torch.tensor([[1.0e-04,  0, 0], [0, 1.0e-06, 0], [0,  0, 9.999999999999998e-11]])
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
dqn = 0.005357142857143

for i in range(n):
    qn[i][2] = dqn * (i % 20)
    qn[i][1] = dqn * math.trunc(i / 20)

