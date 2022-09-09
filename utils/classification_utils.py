import numpy as np
import torch
import gpytorch
import pandas as pd
import matplotlib.pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from sklearn.cluster import KMeans
from sklearn import preprocessing

######## METRICS ##########
def get_accuracy(y_pred, y_true):
    """
    y_pred: predicted probabilities, assumes 2D tensor of shape #samples x #features
    y_true: ground truth, assumes a 1D tensor of shape #samples
    returns the accuracy of the predicted probabilities
    """
    
    max_pred = torch.argmax(y_pred, 1)
    acc = torch.mean((max_pred==y_true).float())
    return(acc)
    
def mean_neg_log_likelihood(y_pred, y_true):
    """
    y_pred: predicted probabilities, assumes 2D tensor of shape #samples x #features
    y_true: ground truth, assumes a 1D tensor of shape #samples
    returns the mean NLL of the predicted probabilities given the true labels
    """
    cat = -torch.distributions.categorical.Categorical(y_pred).log_prob(y_true)
    return(cat.mean())

def calibration_test(y_pred, y_true, nbins=10):
    '''
    COPIED FROM DGP PAPER
    y_pred: predicted probabilities, assumes 2D tensor of shape #samples x #features
    y_true: ground truth, assumes a 1D tensor of shape #samples
    Returns ece:  Expected Calibration Error
            conf: confindence levels (as many as nbins)
            accu: accuracy for a certain confidence level
                  We are interested in the plot confidence vs accuracy
            bin_sizes: how many points lie within a certain confidence level
    '''
    edges = torch.linspace(0, 1, nbins+1)
    accu = torch.zeros(nbins)
    conf = torch.zeros(nbins)
    bin_sizes = torch.zeros(nbins)
    # Multiclass problems are treated by considering the max
    pred = torch.argmax(y_pred, dim=1)
    prob = torch.max(y_pred, dim=1)[0]
    
    #
    y_true = y_true.view(-1)
    prob = prob.view(-1)
    for i in range(nbins):
        idx_in_bin = (prob > edges[i]) & (prob <= edges[i+1])
        bin_sizes[i] = max(sum(idx_in_bin), 1)
        accu[i] = torch.sum(y_true[idx_in_bin] == pred[idx_in_bin]) / bin_sizes[i]
        conf[i] = (edges[i+1] + edges[i]) / 2
    ece = torch.sum(torch.abs(accu - conf) * bin_sizes) / torch.sum(bin_sizes)
    return ece, conf, accu, bin_sizes


######## DATA PREP ############
def standardise(X):
    scaler = preprocessing.StandardScaler().fit(X)
    X_new = scaler.transform(X)
    return(X_new)

def normalise_minusonetoone(X):
    minx = np.min(X, 0)
    maxx = np.max(X, 0)
    ranges = maxx - minx
    ranges[ranges == 0] = 1 # to avoid NaN
    X = (X - minx) / ranges
    X_new = X * 2 - 1
    return(X_new)

def random_inducing_points(train_x, train_y, num_inducing_points):
    
    inducing_idxs = np.random.choice(len(train_x), num_inducing_points, replace=False)
    train_x_induced = torch.tensor(train_x[inducing_idxs]).float()
    train_y_induced = torch.tensor(train_y[inducing_idxs]).long()
    return(train_x_induced, train_y_induced)

def k_means_inducing_points(train_x, train_y, num_inducing_points):
    
    kmeans = KMeans(n_clusters=num_inducing_points, random_state=0).fit(train_x)
    idxs = []
    for k in kmeans.cluster_centers_:
        i = np.abs(train_x.numpy() - k).sum(1).argmin()
        idxs.append(i)
    
    return(train_x[idxs], train_y[idxs])

def k_means_inducing_points_LM_beta(train_x, train_y, num_inducing_points, alpha_eps=0.01, beta_eps=0.01):
    
    kmeans = KMeans(n_clusters=num_inducing_points, random_state=0).fit(train_x)
    idxs = []
    for k in kmeans.cluster_centers_:
        i = np.abs(train_x.numpy() - k).sum(1).argmin()
        idxs.append(i)
    
    alphas = torch.ones((num_inducing_points,)) * alpha_eps
    betas = torch.ones((num_inducing_points,)) * beta_eps
    for i, idx in enumerate(idxs):
        label = kmeans.predict(train_x[idx].reshape(1,-1))
        label_idxs = (kmeans.labels_ == label).nonzero()[0]
        label_ys = train_y[label_idxs]
        ones = torch.sum(label_ys)
        zeros = len(label_ys) - ones
        alphas[i] += ones
        betas[i] += zeros
        
    return(train_x[idxs], alphas, betas)

def k_means_inducing_points_LM_Dir(train_x, train_y, num_inducing_points, num_classes, alpha_eps=0.001):
    
    kmeans = KMeans(n_clusters=num_inducing_points, random_state=0).fit(train_x)
    idxs = []
    for k in kmeans.cluster_centers_:
        i = np.abs(train_x.numpy() - k).sum(1).argmin()
        idxs.append(i)
    
    alphas = torch.ones((num_inducing_points, num_classes)) * alpha_eps
    for i, idx in enumerate(idxs):
        label = kmeans.predict(train_x[idx].reshape(1,-1))
        label_idxs = (kmeans.labels_ == label).nonzero()[0]
        label_ys = train_y[label_idxs]
        all_classes = torch.nn.functional.one_hot(label_ys, num_classes).sum(0) 
        alphas[i] += all_classes
        
    return(train_x[idxs], alphas)

def plot_res(res):
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    ax[0].plot(res["lengthscales"], res["accuracy_results"])
    ax[0].set_xscale('log')
    ax[0].set_title("accuracy")
    ax[0].set_xticks(res["lengthscales"])
    ax[0].set_xticklabels(res["lengthscales"])

    
    ax[1].plot(res["lengthscales"], res["nll_results"])
    ax[1].set_xscale('log')
    ax[1].set_title("mean NLL")
    ax[1].set_xticks(res["lengthscales"])
    ax[1].set_xticklabels(res["lengthscales"])

    
    ax[2].plot(res["lengthscales"], res["ece_results"])
    ax[2].set_xscale('log')
    ax[2].set_title("ECE")
    ax[2].set_xticks(res["lengthscales"])
    ax[2].set_xticklabels(res["lengthscales"])

    
    plt.show()



