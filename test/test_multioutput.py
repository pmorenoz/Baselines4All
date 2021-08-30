

#
# Pablo Moreno-Munoz (pabmo@dtu.dk)
# Technical University of Denmark - DTU
# May 2021

import torch
import numpy as np

from util import smooth_function

from kernels.rbf import RBF
from likelihoods.gaussian import Gaussian
from models.svgp import SVGP
from models.svmogp import SVMOGP
from models.ensemblegp import EnsembleGP
from models.moensemble import MultiOutputEnsembleGP
#from models.svgp import predictive
from optimization.algorithms import vem_algorithm, ensemble_vem, moensemble_vem

from util import DataGP, DataMOGP
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

###########################
#                         #
#    SYNTHETIC DATA       #
#                         #
###########################

Q = 2
N_1 = 5000
N_2 = 5000
x_1,_ = torch.sort(torch.rand(N_1,1), 0)
x_2,_ = torch.sort(torch.rand(N_2,1), 0)

x = [x_1, x_2]

# True U functions
def true_u_functions(X_list):
    u_functions = []
    for X in X_list:
        u_task = torch.empty(X.shape[0], 2)
        u_task[:, 0, None] = 4.5 * torch.cos(2 * np.pi * X + 1.5 * np.pi) - \
                             3 * torch.sin(4.3 * np.pi * X + 0.3 * np.pi) + \
                             5 * torch.cos(7 * np.pi * X + 2.4 * np.pi)
        u_task[:, 1, None] = 4.5 * torch.cos(1.5 * np.pi * X + 0.5 * np.pi) + \
                             5 * torch.sin(3 * np.pi * X + 1.5 * np.pi) - \
                             5.5 * torch.cos(8 * np.pi * X + 0.25 * np.pi)

        u_functions.append(u_task)
    return u_functions

# True F functions
def true_f_functions(true_u, X_list):
    true_f = []
    W = W_lincombination()
    # D=1
    for d in range(2):
        f_d = torch.zeros(X_list[d].shape[0], 1)
        for q in range(2):
            f_d += W[q][d].T * true_u[d][:, q, None]
        true_f.append(f_d)

    return true_f, W

# True W combinations
def W_lincombination():
    W_list = []
    # q=1
    Wq1 = torch.tensor(([[-0.5], [0.1]]))
    W_list.append(Wq1)
    # q=2
    Wq2 = torch.tensor(([[-0.1], [.6]]))
    W_list.append(Wq2)
    return W_list

# True functions values for inputs X
trueU = true_u_functions(x)
[f_1, f_2], _ = true_f_functions(trueU, x)

y_1 = f_1 + 0.5*torch.randn(N_1, 1)
y_2 = f_2 + 0.5*torch.randn(N_2, 1)

y = [y_1, y_2]

###########################
#                         #
#         MODEL           #
#                         #
###########################

num_epochs = 10
batch_size = 50
batch_rates = []
for x_d in x:
    br_d = len(x_d)/batch_size
    batch_rates.append(br_d)

data_torch = DataMOGP(x=x, y=y)
data_loader = DataLoader(data_torch, batch_size=batch_size, shuffle=True)

D = len(y)
M = 10
Q = 3
kernels = Q*[RBF()]
likelihoods = [Gaussian(), Gaussian()]
model = SVMOGP(kernels=kernels, likelihoods=likelihoods, Q=Q, M=M, batch_rates=batch_rates)

# OPTIMIZATION -- -- --
lr_m = 1e-1
lr_L = 1e-2
lr_B = 1e-2
lr_hyp = 1e-3
lr_z = 1e-3

optimizer = torch.optim.Adam([{'params': model.q_m, 'lr': lr_m},
                             {'params': model.q_L, 'lr': lr_L},
                              {'params': model.kernels.parameters(), 'lr': lr_hyp},
                              {'params': model.coregionalization.W, 'lr': lr_B},
                              {'params': model.z, 'lr': lr_z}], lr=0.1)

elbo_curve = []
for epoch in range(num_epochs):
    size = len(data_loader.dataset)
    for batch, (x,y) in enumerate(data_loader):

        #x = Variable(x).float()
        #y = Variable(y).float()

        elbo_it = model(x, y)
        optimizer.zero_grad()
        elbo_it.backward()  # Backward pass <- computes gradients
        optimizer.step()

        elbo_curve.append(-model(x, y).item())

        if batch % 50 == 0:
            elbo_it, current = elbo_it.item(), batch * len(x)
            print(f"ELBO: {elbo_it:>7f} // epoch:{epoch:>2d} [{current:>5d}/{size:>5d}]")


plt.figure()
plt.plot(elbo_curve)
plt.show()

###########################
#                         #
#        PREDICTION       #
#                         #
###########################


# Test data
N_test = 200
x_test = torch.linspace(0, 1.0, N_test)[:,None]
gp, gp_upper, gp_lower = model.predictive(x_test, d=0)

gp_lower = gp_lower - 2*model.likelihoods[0].sigma.detach().numpy()
gp_upper = gp_upper + 2*model.likelihoods[0].sigma.detach().numpy()

# Plot Ensemble
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(x_1,y_1,'bx')

plt.plot(x_test, gp, 'k-')
plt.plot(x_test, gp_upper, 'k-')
plt.plot(x_test, gp_lower, 'k-')
plt.xlim([0,1.0])

plt.title('Multi-output Ensemble GP Model -- (M = 7)')
plt.xlabel('Input X')
plt.ylabel('Output Function')

gp, gp_upper, gp_lower = model.predictive(x_test, d=1)
gp_lower = gp_lower - 2*model.likelihoods[1].sigma.detach().numpy()
gp_upper = gp_upper + 2*model.likelihoods[1].sigma.detach().numpy()

plt.subplot(2,1,2)
plt.plot(x_2,y_2,'rx')

plt.plot(x_test, gp, 'k-')
plt.plot(x_test, gp_upper, 'k-')
plt.plot(x_test, gp_lower, 'k-')
plt.xlim([0.0,1.0])

plt.xlabel('Input X')
plt.ylabel('Output Function')

plt.show()