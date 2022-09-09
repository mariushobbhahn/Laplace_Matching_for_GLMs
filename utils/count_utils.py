import numpy as np
import torch
import gpytorch
import pandas as pd
import matplotlib.pyplot as plt


def get_RMSE(pred, true):
    return(torch.sqrt(torch.mean((pred - true)**2)))

def get_mean_neg_log_likelihood_poisson(pred, true):
    return(-torch.distributions.poisson.Poisson(pred).log_prob(true).mean())

def get_in_2_std(lb, ub, test_y):
    bigger_than_lower = test_y > lb
    smaller_than_upper = test_y < ub
    between = torch.logical_and(bigger_than_lower, smaller_than_upper).long()
    return(between.float().mean())


def plot_res(res):
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    ax[0].plot(res["lengthscales"], res["rmse_results"])
    ax[0].set_xscale('log')
    ax[0].set_title("RMSE")
    ax[0].set_xticks(res["lengthscales"])
    ax[0].set_xticklabels(res["lengthscales"])

    
    ax[1].plot(res["lengthscales"], res["nll_results"])
    ax[1].set_xscale('log')
    ax[1].set_title("mean NLL")
    ax[1].set_xticks(res["lengthscales"])
    ax[1].set_xticklabels(res["lengthscales"])

    
    ax[2].plot(res["lengthscales"], res["in2std_results"])
    ax[2].set_xscale('log')
    ax[2].set_title("in 2 std")
    ax[2].set_xticks(res["lengthscales"])
    ax[2].set_xticklabels(res["lengthscales"])

    
    plt.show()