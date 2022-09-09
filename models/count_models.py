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

######### Laplace Matching + Gamma ##########

# translate pseudo counts to Gaussians via LM
def transform_y_Gamma_LM(y_train_induced, alpha_eps=0.1, counts=1):
    alphas = y_train_induced + alpha_eps
    betas = torch.ones_like(alphas) * 1/counts # <- because we only have one data point per timestep
    train_mu_LM = np.log(alphas/betas)
    train_var_LM = 1/alphas
    return(train_mu_LM, train_var_LM)


def create_LM_Gamma_GP_model(train_x, train_y_mu, train_y_var, init_lengthscale=1, kernel="RBF",
                            learn_additional_noise=True, fixed_likelihood=True):
    
    assert kernel in ["RBF", "RQ"]
    
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            if kernel == "RBF":
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            elif kernel == "RQ":
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
            self.covar_module.base_kernel.lengthscale = init_lengthscale

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    if fixed_likelihood:
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.tensor(np.sqrt(train_y_var)), 
                                                        learn_additional_noise=learn_additional_noise)
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_covar=torch.tensor(np.sqrt(train_y_var)), 
                                                        learn_additional_noise=learn_additional_noise)
    
    model = ExactGPModel(train_x, train_y_mu, likelihood)
    return(model, likelihood)

def train_LM_Gamma_GP_model(train_x, train_y_mu, model, likelihood, num_iter=500, lr=0.1, report_iter=50):
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(num_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y_mu)
        loss.backward()
        if i % report_iter == 0:
            print('Iter %d/%d - Loss: %.3f; lengthscale: %.3f; noise: %.3f' % (
                i + 1, num_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.mean().item()
            ))
        optimizer.step()
        
    return(model, likelihood)

def evaluate_LM_Gamma_GP(model, likelihood, test_x, test_y, num_samples=1000, fixed_likelihood=False):
    
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if fixed_likelihood:
            pred_dist = likelihood(model(test_x), noise=torch.mean(likelihood.noise))
        else:
            pred_dist = likelihood(model(test_x))

    pred_samples = pred_dist.sample(torch.Size((num_samples,)))
    pred_samples_exp = pred_samples.exp()
    pred_mean = pred_samples_exp.mean(0)
    
    lb, ub = pred_dist.confidence_region()
    lb, ub = torch.exp(lb), torch.exp(ub)
    
    # evaluate
    rmse = count_utils.get_RMSE(pred_mean, test_y)
    mnll = count_utils.get_mean_neg_log_likelihood_poisson(pred_mean, test_y)
    in_2_std = count_utils.get_in_2_std(lb, ub, test_y)

    return(rmse.item(), mnll.item(), in_2_std.item())

######### SVIGP ################
def create_SVIGP_model(train_x, init_lengthscale=1, num_inducing_points=500, kernel="RBF"):
    
    assert kernel in ["RBF", "RQ"]
    
    class GPModel(ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
            super(GPModel, self).__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            if kernel == "RBF":
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            elif kernel == "RQ":
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
            self.covar_module.base_kernel.lengthscale = init_lengthscale

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    inducing_points = train_x[:num_inducing_points].float()
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    return(model, likelihood)

def train_SVIGP_model(train_loader, train_y, model, likelihood, num_iter=500, lr=0.1, report_iter=50):
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    epochs_iter = tqdm.notebook.tqdm(range(num_iter), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        batch_loss = 0
        minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            batch_loss += loss.item()
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
        
        if i % report_iter == 0:
                print("iter: {}/{}".format(i, num_iter))
                print("loss: ", batch_loss)
                print("lengthscale: ", model.covar_module.base_kernel.lengthscale.item())
        
    return(model, likelihood)

def evaluate_SVIGP(model, likelihood, test_loader, test_y, greater_zero_constraint=True):
    
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    lbs, ubs = torch.tensor([0.]), torch.tensor([0.])

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
            lb, ub = preds.confidence_region()
            lbs = torch.cat([lbs, lb])
            ubs = torch.cat([ubs, ub])
    pred_mean = means[1:]
    lb, ub = lbs[1:], ubs[1:]

    # all values have to be positive
    if greater_zero_constraint:
        pred_mean = torch.maximum(pred_mean, 1e-6*torch.ones_like(pred_mean))
    
    # evaluate
    rmse = count_utils.get_RMSE(pred_mean, test_y)
    mnll = count_utils.get_mean_neg_log_likelihood_poisson(pred_mean, test_y)
    in_2_std = count_utils.get_in_2_std(lb, ub, test_y)

    return(rmse.item(), mnll.item(), in_2_std.item())


######### ExactGP #############
def create_ExactGP_model(train_x, train_y, init_lengthscale=1, kernel="RBF"):
    
    assert kernel in ["RBF", "RQ"]
    
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            if kernel == "RBF":
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            elif kernel == "RQ":
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
            self.covar_module.base_kernel.lengthscale = init_lengthscale

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    likelihood = gpytorch.likelihoods.GaussianLikelihood(train_additional_noise=True)
    model = ExactGPModel(train_x, train_y, likelihood)
    return(model, likelihood)

def train_ExactGP_model(train_x, train_y, model, likelihood, num_iter=500, lr=0.1, report_iter=50):
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(num_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print(model.covar_module.base_kernel)
        if i % report_iter == 0:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
                i + 1, num_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item()
            ))
        optimizer.step()
        
    return(model, likelihood)

def evaluate_ExactGP(model, likelihood, test_x, test_y, greater_zero_constraint=True):
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds_exact = likelihood(model(test_x))
        
    pred_mean = preds_exact.mean
    lb, ub = preds_exact.confidence_region()
    if greater_zero_constraint:
        pred_mean = torch.maximum(pred_mean, 1e-6*torch.ones_like(pred_mean))
    
    # evaluate
    rmse = count_utils.get_RMSE(pred_mean, test_y)
    mnll = count_utils.get_mean_neg_log_likelihood_poisson(pred_mean, test_y)
    in_2_std = count_utils.get_in_2_std(lb, ub, test_y)

    return(rmse.item(), mnll.item(), in_2_std.item())

######### SVIGP+log transform ######
def evaluate_SVIGP_log(model, likelihood, test_loader, test_y, greater_zero_constraint=True):
    
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    lbs, ubs = torch.tensor([0.]), torch.tensor([0.])

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
            lb, ub = preds.confidence_region()
            lbs = torch.cat([lbs, lb])
            ubs = torch.cat([ubs, ub])
    
    pred_mean = torch.exp(means[1:])
    lb, ub = torch.exp(lbs[1:]), torch.exp(ubs[1:])

    # all values have to be positive
    if greater_zero_constraint:
        pred_mean = torch.maximum(pred_mean, 1e-6*torch.ones_like(pred_mean))
    
    # evaluate
    rmse = count_utils.get_RMSE(pred_mean, test_y)
    mnll = count_utils.get_mean_neg_log_likelihood_poisson(pred_mean, test_y)
    in_2_std = count_utils.get_in_2_std(lb, ub, test_y)

    return(rmse.item(), mnll.item(), in_2_std.item())

######### ExactGP+log transform ###########
def evaluate_ExactGP_log(model, likelihood, test_x, test_y, greater_zero_constraint=True):
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds_exact = likelihood(model(test_x))
        
    pred_mean = torch.exp(preds_exact.mean)
    lb, ub = preds_exact.confidence_region()
    lb, ub = torch.exp(lb), torch.exp(ub)
    if greater_zero_constraint:
        pred_mean = torch.maximum(pred_mean, 1e-6*torch.ones_like(pred_mean))
    
    # evaluate
    rmse = count_utils.get_RMSE(pred_mean, test_y)
    mnll = count_utils.get_mean_neg_log_likelihood_poisson(pred_mean, test_y)
    in_2_std = count_utils.get_in_2_std(lb, ub, test_y)

    return(rmse.item(), mnll.item(), in_2_std.item())


######### Cross validation ##########

# find good initial hyperparameters with k-fold cross validation on the training dataset (Dirichlet). 
def select_init_lengthscale_with_CV_Gamma(train_x, train_y, mode="LMGP_gamma", num_inducing_points=200, 
                                    learn_noise=True, num_iter=500, lr=0.1, alpha_eps=0.1, fixed_likelihood=False,
                                    lengthscales=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], kernel="RBF",
                                    max_test_size=1000, num_classes=10):
    
    assert mode in ["ExactGP", "ExactGP_log", "LMGP_gamma", "SVIGP", "SVIGP_log"]


    #outer loop
    rmse_results = []
    nll_results = []
    in2std_results = []

    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for i_, (train_index, test_index) in enumerate(kf.split(train_x)):
        print("fold: {}".format(i_))
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        X_test, y_test = X_test[:max_test_size], y_test[:max_test_size]
        
        # Create inducing points
        # TODO
        X_train_induced = X_train
        y_train_induced = y_train
        
        
        # make train loader for SVI
        if mode in ["SVIGP", "SVIGP_log"]:
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

            train_dataset_log = TensorDataset(X_train, torch.log(torch.maximum(y_train, 0.1*torch.ones_like(y_train))))
            train_loader_log = DataLoader(train_dataset_log, batch_size=1024, shuffle=True)

            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        
        rmse_results_inner = []
        nll_results_inner = []
        in2std_results_inner = []
        for l in lengthscales:
            
            print("current lengthscale: {}".format(l))
            
            # train the model on inducing points
            if mode == "ExactGP":
                ExactGP_model, ExactGP_likelihood = create_ExactGP_model(X_train_induced, y_train_induced, 
                                                                     init_lengthscale=l, kernel=kernel)
                ExactGP_model, ExactGP_likelihood = train_ExactGP_model(X_train_induced, y_train_induced, ExactGP_model, 
                                                            ExactGP_likelihood, num_iter=num_iter, lr=lr, report_iter=num_iter//10)
                rmse, mnll, in2std = evaluate_ExactGP(ExactGP_model, ExactGP_likelihood, X_test, y_test)

            elif mode == "ExactGP_log":
                y_train_induced = torch.log(torch.maximum(y_train_induced, 0.1*torch.ones_like(y_train_induced)))
                ExactGP_model_log, ExactGP_likelihood_log = create_ExactGP_model(X_train_induced, y_train_induced, 
                                                                                init_lengthscale=l, kernel=kernel)
                ExactGP_model_log, ExactGP_likelihood_log = train_ExactGP_model(X_train_induced, y_train_induced, 
                                                ExactGP_model_log, ExactGP_likelihood_log, num_iter=num_iter, lr=lr, report_iter=num_iter//10)
                rmse, mnll, in2std = evaluate_ExactGP_log(ExactGP_model_log, ExactGP_likelihood_log, X_test, y_test)
                
            elif mode == "LMGP_gamma":
                y_train_induced_mu, y_train_induced_var = transform_y_Gamma_LM(y_train_induced, 
                                                                            alpha_eps=alpha_eps, counts=1)
                LMGP_model, LMGP_likelihood = create_LM_Gamma_GP_model(X_train_induced, y_train_induced_mu, y_train_induced_var,
                                                    kernel=kernel, init_lengthscale=l, fixed_likelihood=fixed_likelihood)
                LMGP_model, LMGP_likelihood = train_LM_Gamma_GP_model(X_train_induced, y_train_induced_mu,
                                                LMGP_model, LMGP_likelihood, num_iter=num_iter, lr=lr, report_iter=num_iter//10)
                rmse, mnll, in2std = evaluate_LM_Gamma_GP(LMGP_model, LMGP_likelihood, X_test, y_test)
                
            elif mode == "SVIGP":
                model_SVI, likelihood_SVI = create_SVIGP_model(X_train, kernel=kernel,
                                                        init_lengthscale=l, num_inducing_points=num_inducing_points)
                model_SVI, likelihood_SVI = train_SVIGP_model(train_loader, y_train_induced, model_SVI, likelihood_SVI,
                                                num_iter=num_iter, lr=lr, report_iter=num_iter//10)
                rmse, mnll, in2std = evaluate_SVIGP(model_SVI, likelihood_SVI, test_loader, y_test)
                
            elif mode == "SVIGP_log":
                y_train_induced = torch.log(torch.maximum(y_train_induced, 0.1*torch.ones_like(y_train_induced)))
                model_SVI_log, likelihood_SVI_log = create_SVIGP_model(X_train, kernel=kernel, 
                                                        init_lengthscale=l, num_inducing_points=num_inducing_points)
                model_SVI_log, likelihood_SVI_log = train_SVIGP_model(train_loader_log, y_train_induced, 
                                                                    model_SVI_log, likelihood_SVI_log,
                                                num_iter=num_iter, lr=lr, report_iter=num_iter//10)
                rmse, mnll, in2std = evaluate_SVIGP_log(model_SVI_log, likelihood_SVI_log, test_loader, y_test)
                
            
            # print current results:
            print("RMSE: {};\t mnll:  {}; \t in2std: {}".format(rmse, mnll, in2std))
            rmse_results_inner.append(rmse)
            nll_results_inner.append(mnll)
            in2std_results_inner.append(in2std)

        rmse_results.append(rmse_results_inner)
        nll_results.append(nll_results_inner)
        in2std_results.append(in2std_results_inner)
        
    rmse_results = np.mean(rmse_results, 0)
    nll_results = np.mean(nll_results, 0)
    in2std_results = np.mean(in2std_results, 0)

    res = {
        "lengthscales":lengthscales,
        "rmse_results":rmse_results, 
        "nll_results":nll_results,
        "in2std_results":in2std_results
    }
    
    return(res)

