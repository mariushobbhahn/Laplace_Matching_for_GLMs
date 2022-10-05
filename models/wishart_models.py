import math
import torch
import numpy as np
import pandas as pd
import gpytorch
from matplotlib import pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import tqdm
from sklearn.model_selection import KFold

import count_utils

######### Laplace Matching + Wishart ##########
# transform all ns and Psis to mus and Sigmas through Laplace Matching
from scipy.linalg import expm, sqrtm, logm

def Inv_Wishart_logm_bridge_mu(Psi, v):
    p = np.shape(Psi)[0]
    r = logm(1/(v+p-1) * Psi)
    return(r.reshape(-1))
    
def Inv_Wishart_logm_bridge_Sigma(Psi, v):
    p = np.shape(Psi)[0]
    return(2 * (v+p-1) * np.eye(p**2))

def Inv_Wishart_sqrtm_bridge_mu(Psi, v):
    p = np.shape(Psi)[0]
    r =  sqrtm(1/(v+p)*Psi)
    return(r.reshape(-1))
    
def Inv_Wishart_sqrtm_bridge_Sigma(Psi, v):
    p = np.shape(Psi)[0]
    Psi_inv = np.linalg.inv(Psi)
    Psi_sqrtm = sqrtm(Psi)
    Psi_inv_sqrtm = np.linalg.inv(Psi_sqrtm)
    one = (v+p)**2 * (np.kron(Psi_inv_sqrtm@Psi_inv_sqrtm@Psi_inv_sqrtm, Psi_sqrtm)\
                         + np.kron(Psi_inv, np.eye(p)))

    
    R = np.linalg.inv(one)
    return(R) 


#helper functions
def is_pos_def(x):
    eig = np.linalg.eigvals(x)
    pd = np.all(eig > 0)
    if not pd:
        raise(ValueError("matrix is not positive definite! Matrix: {}; Eigenvalues: {}".format(x, eig)))
    else:
        return(True)