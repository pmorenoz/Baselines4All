

from kernels.rbf import RBF
from likelihoods.gaussian import Gaussian
from likelihoods.bernoulli import Bernoulli
from models.svgp import SVGP
from models.ensemblegp import EnsembleGP
from optimization.algorithms import vem_algorithm, ensemble_vem, ensemble_vem_parallel
from optimization.algorithms import AlgorithmVEM

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save

from util import DataGP
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# COOLORS.CO palettes
color_palette_1 = ['#335c67','#fff3b0','#e09f3e','#9e2a2b','#540b0e']
color_palette_2 = ['#177e89','#084c61','#db3a34','#ef8354','#323031']
color_palette_3 = ['#bce784','#5dd39e','#348aa7','#525274','#513b56']
color_palette_4 = ['#002642','#840032','#e59500','#e5dada','#02040e']
color_palette_5 = ['#202c39','#283845','#b8b08d','#f2d449','#f29559']
color_palette_6 = ['#21295c','#1b3b6f','#065a82','#1c7293','#9eb3c2']
color_palette_7 = ['#f7b267','#f79d65','#f4845f','#f27059','#f25c54']

color_palette = color_palette_2
color_1 = color_palette_3[4]
color_0 = color_palette_2[2]

print('')
print('-- DATA LOADING -----------------------')
print('')

# Load Data --
#data = sio.loadmat('../data/london.mat')
data = sio.loadmat('../data/london_r.mat')
y_real = data['Yprice']
y_bin = data['Ycontract']
x = data['X']

#greater london limits:
#latitude (North-South) ~55 is Y axis
#longitude (East-West) ~0 is X axis
xmin_london = -0.5105
xmax_london = 0.3336
ymin_london = 51.2871
ymax_london = 51.6925

#dim 0 of X -> yaxis
#dim 1 of X -> xaxis
x[:,1] = (x[:,1] - xmin_london)/(xmax_london - xmin_london)
x[:,0] = (x[:,0] - ymin_london)/(ymax_london - ymin_london)

y = y_bin

print('')
print('-- CLASSIFICATION -----------------------')
print('')

num_epochs = 20
batch_size = 500
N_test = 100
M_c = 6

data_torch = DataGP(x=x, y=y)
data_loader = DataLoader(data_torch, batch_size=batch_size, shuffle=True)

batch_rate = float(len(data_loader.dataset))/batch_size
kernel_c = RBF()
likelihood_c = Bernoulli()
model = SVGP(kernel_c, likelihood_c, M_c**2, input_dim=2, batch_rate=batch_rate)

# initial grid of inducing-points
mx = np.mean(x[:, 0])
my = np.mean(x[:, 1])
vx = np.var(x[:, 0])
vy = np.var(x[:, 1])

zy = np.linspace(my - 0.2, my + 0.3, M_c)
zx = np.linspace(mx - 0.3, mx + 0.2, M_c)

ZX, ZY = np.meshgrid(zx, zy)
ZX = ZX.reshape(M_c ** 2, 1)
ZY = ZY.reshape(M_c ** 2, 1)
Z = np.hstack((ZX, ZY))
z_c = torch.from_numpy(Z).float()

model.z = torch.nn.Parameter(z_c, requires_grad=True)

# OPTIMIZATION -- -- --
lr_m = 1e-1
lr_L = 1e-2
lr_hyp = 1e-4
lr_z = 1e-3

optimizer = torch.optim.Adam([{'params': model.q_m, 'lr': lr_m},
                             {'params': model.q_L, 'lr': lr_L},
                              {'params': model.kernel.parameters(), 'lr': lr_hyp},
                              {'params': model.z, 'lr': lr_z}], lr=0.1)


elbo_curve = []
for epoch in range(num_epochs):
    size = len(data_loader.dataset)
    for batch, (x,y) in enumerate(data_loader):

        x = Variable(x).float()
        y = Variable(y).float()

        elbo_it = model(x, y)
        optimizer.zero_grad()
        elbo_it.backward()  # Backward pass <- computes gradients
        optimizer.step()

        elbo_curve.append(-model(x, y).item())

        if batch % 50 == 0:
            elbo_it, current = elbo_it.item(), batch * len(x)
            print(f"ELBO: {elbo_it:>7f} - epoch: {epoch:>2d} [{current:>5d}/{size:>5d}]")


plt.figure()
plt.plot(elbo_curve)
plt.show()

print('')
print('-- PREDICTION AND PLOT -----------------------')
print('')

x = data['X']
y = y_bin

sigmoid = torch.nn.Sigmoid()

min_tx = 0.0
min_ty = 0.0
max_tx = 1.0
max_ty = 1.0

ty = np.linspace(min_ty, max_ty, N_test)
tx = np.linspace(min_tx, max_tx, N_test)
TX_grid, TY_grid = np.meshgrid(tx, ty)
TX = TX_grid.reshape(N_test ** 2, 1)
TY = TY_grid.reshape(N_test ** 2, 1)
X_test = np.hstack((TX, TY))
x_test = torch.from_numpy(X_test).float()

gp, gp_upper, gp_lower = model.predictive(x_test)
gp = sigmoid(torch.from_numpy(gp))

# Plot
plt.figure(figsize=(10, 8))
ax = plt.axes()
plt.plot(x[y[:, 0] == 0, 1], x[y[:, 0] == 0, 0], ls='', marker='o', color=color_0, alpha=0.5, ms=5.0)
plt.plot(x[y[:, 0] == 1, 1], x[y[:, 0] == 1, 0], ls='', marker='o', color=color_1, alpha=0.5, ms=5.0)
plt.plot(model.z[:, 0].detach(), model.z[:, 1].detach(), 'kx', ms=10.0, mew=2.0)
cs = ax.contour(TX_grid, TY_grid, np.reshape(gp, (N_test, N_test)), linewidths=3, colors='k',
                levels=[0.2, 0.3, 0.5, 0.7, 0.8], zorder=10)
ax.clabel(cs, inline=1, fontsize=14, fmt='%1.1f')

# true location -- labels
xdif = xmax_london - xmin_london
xlabels = [str(float("{0:.2f}".format(xmin_london))), str(float("{0:.2f}".format(xmin_london + (xdif / 5)))),
           str(float("{0:.2f}".format(xmin_london + (2 * xdif / 5)))),
           str(float("{0:.2f}".format(xmin_london + (3 * xdif / 5)))),
           str(float("{0:.2f}".format(xmin_london + (4 * xdif / 5)))),
           str(float("{0:.2f}".format(xmax_london)))]
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabels)
ydif = ymax_london - ymin_london
ylabels = [str(float("{0:.2f}".format(ymin_london))), str(float("{0:.2f}".format(ymin_london + (ydif / 5)))),
           str(float("{0:.2f}".format(ymin_london + (2 * ydif / 5)))),
           str(float("{0:.2f}".format(ymin_london + (3 * ydif / 5)))),
           str(float("{0:.2f}".format(ymin_london + (4 * ydif / 5)))),
           str(float("{0:.2f}".format(ymax_london)))]
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ylabels)

plt.plot(r'Type of contract -- Classification task')
plt.ylabel(r'Latitude ($x_1$)')
plt.xlabel(r'Longitude ($x_2$)')

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)

save = False
if save:
    trial = 0
    plt.savefig(fname='./figs/heterogeneous/trial_'+str(trial)+'_classification.pdf', format='pdf')
    torch.save(model.state_dict(), './files/heterogeneous/classification.pt')

    plt.close()

plt.show()