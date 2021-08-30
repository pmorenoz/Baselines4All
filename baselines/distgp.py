# Implementation of the "Distributed GP"
# by Deisenroth & Ng, ICML 2015
#
# Pablo Moreno-Munoz (pmoreno@tsc.uc3m.es)
# Universidad Carlos III de Madrid
# April 2020

import torch

class DistGP(torch.nn.Module):
    """
    -- Distributed Gaussian Process Regression--
    """
    def __init__(self, kernel, likelihood, input_dim=None):
        super(DistGP, self).__init__()

        if input_dim is None:
            input_dim = 1

        self.input_dim = int(input_dim)     #dimension of x

        # GP Elements --
        self.likelihood = likelihood        #type of likelihood
        self.kernel = kernel                #type of kernel


    def forward(self, x, y):
        identity = torch.eye(y.size(0))
        s_n = torch.pow(self.likelihood.sigma, 2.0)

        K = self.kernel.K(x,x)
        KI = K + torch.mul(s_n,identity)
        iKI, _ = torch.solve(torch.eye(KI.size(0)), KI)
        yiKIy = y.t().mm(iKI).mm(y)

        log_marginal = -0.5*yiKIy - 0.5*torch.logdet(KI)
        return -log_marginal

    def predictive(self, x, y, x_new):

        Kx = self.kernel.K(x, x_new)
        Kxx = self.kernel.K(x_new, x_new)

        identity = torch.eye(y.size(0))
        s_n = torch.pow(self.likelihood.sigma, 2.0)

        K = self.kernel.K(x, x)
        KI = K + torch.mul(s_n, identity)
        iKI, _ = torch.solve(torch.eye(KI.size(0)), KI)

        gp_m = Kx.t().mm(iKI).mm(y)
        gp_v = torch.diagonal(Kxx - Kx.t().mm(iKI).mm(Kx), 0)[:,None]

        return gp_m, gp_v

    def rmse(self, x, y, x_new, f_new):
        f_gp,_ = self.predictive(x, y, x_new)
        rmse = torch.sqrt(torch.mean((f_new - f_gp)**2.0)).detach().numpy()
        return rmse

    def mae(self, x, y, x_new, f_new):
        f_gp,_ = self.predictive(x, y, x_new)
        mae = torch.mean(torch.abs(f_new - f_gp)).detach().numpy()
        return mae

    def nlpd(self, x, y, x_new, y_new):
        f_gp, u_gp = self.predictive(x, y, x_new)
        #f_gp = torch.from_numpy(f_gp)
        #u_gp = torch.from_numpy(u_gp)
        v_gp = torch.pow(0.5*(u_gp - f_gp), 2.0)
        nlpd = - torch.mean(self.likelihood.log_predictive(y_new, f_gp, v_gp)).detach().numpy()
        return nlpd
