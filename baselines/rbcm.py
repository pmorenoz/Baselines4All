# Implementation of the "Robust Bayesian Committee Machine"
# by Deisenroth & Ng, ICML 2015
#
# Pablo Moreno-Munoz (pmoreno@tsc.uc3m.es)
# Universidad Carlos III de Madrid
# May 2020

import torch

class RobustBayesianCM(torch.nn.Module):
    """
    -- Robust Bayesian Committee Machine --
    """
    def __init__(self, models, input_dim=1.0):
        super(RobustBayesianCM, self).__init__()

        self.input_dim = int(input_dim)  # dimension of x

        # Adjacent Local GP Models
        self.models = models  # is a list

    def forward(self):
        return 1.0

    def predictive(self, x, y, x_new):
        # x is a list of x_k (distributed)
        # y is a list of y_k (distributed)

        K = len(self.models)
        prior_v = torch.diagonal(self.models[0].kernel.K(x_new, x_new), 0)[:, None]
        log_prior_v = torch.log(prior_v)

        gp_m = torch.zeros(x_new.size())
        gp_v = torch.zeros(x_new.size())
        correction = torch.zeros(x_new.size())

        for k, model_k in enumerate(self.models):
            m_k, v_k = model_k.predictive(x[k], y[k], x_new)

            beta_k = 0.5*(log_prior_v - torch.log(v_k))
            correction += beta_k

            gp_m += m_k/v_k
            gp_v += 1.0/v_k

        correction = (1-correction)/prior_v
        gp_v = 1.0/(gp_v + correction)
        gp_m = gp_v*gp_m

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
        f_gp, v_gp = self.predictive(x, y, x_new)
        nlpd = - torch.mean(self.models[0].likelihood.log_predictive(y_new, f_gp, v_gp)).detach().numpy()
        return nlpd

    # FOR HIERARCHICAL SETTINGS

    def predictive_layer(self, gps_m, gps_v, x_new):
        # gps_m is a list of gp_m (distributed)
        # gps_v is a list of gp_v (distributed)

        K = len(gps_m)
        prior_v = torch.diagonal(self.models[0].kernel.K(x_new, x_new), 0)[:, None]
        log_prior_v = torch.log(prior_v)
        log_prior_v[torch.isnan(log_prior_v)] = 1e-3

        gp_m = torch.zeros(x_new.size())
        gp_v = torch.zeros(x_new.size())
        correction = torch.zeros(x_new.size())

        for k, m_k in enumerate(gps_m):
            v_k = gps_v[k]

            log_v_k = torch.log(v_k)
            log_v_k[torch.isnan(log_v_k)] = 1e-3

            beta_k = 0.5 * (log_prior_v - log_v_k)
            correction += beta_k

            gp_m += m_k / v_k
            gp_v += 1.0 / v_k

        correction = (1 - correction) / prior_v
        gp_v = 1.0 / (gp_v + correction)
        gp_m = gp_v * gp_m

        return gp_m, gp_v

    def rmse_layer(self, gps_m, gps_v, x_new, f_new):
        f_gp,_ = self.predictive_layer(gps_m, gps_v, x_new)
        rmse = torch.sqrt(torch.mean((f_new - f_gp)**2.0)).detach().numpy()
        return rmse

    def mae_layer(self, gps_m, gps_v, x_new, f_new):
        f_gp,_ = self.predictive_layer(gps_m, gps_v, x_new)
        mae = torch.mean(torch.abs(f_new - f_gp)).detach().numpy()
        return mae

    def nlpd_layer(self, gps_m, gps_v, x_new, y_new):
        f_gp, v_gp = self.predictive_layer(gps_m, gps_v, x_new)
        nlpd = - torch.mean(self.models[0].likelihood.log_predictive(y_new, f_gp, v_gp)).detach().numpy()
        return nlpd