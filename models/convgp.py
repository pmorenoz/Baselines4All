#
# Pablo Moreno-Munoz (pabmo@dtu.dk)
# Technical University of Denmark - DTU
# April 2021

import torch
from torch.distributions import MultivariateNormal as Normal
from torch.distributions import kl_divergence

import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior


class ConvGP(torch.nn.Module):
    """
    -- Convolutional Gaussian Processes --
    """