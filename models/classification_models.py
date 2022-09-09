import numpy as np
import torch
import gpytorch
import pandas as pd
import matplotlib.pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from sklearn.model_selection import KFold

import classification_utils

######### GPC ############
def create_DGP_model(train_x, train_y, learn_additional_noise=True, init_lengthscale=1):

    class DirichletGPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood, num_classes, init_lengthscale=init_lengthscale):
            super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=torch.Size((num_classes,))),
                batch_shape=torch.Size((num_classes,)),
            )
            self.covar_module.base_kernel.lengthscale = init_lengthscale

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=learn_additional_noise)
    model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, 
                            num_classes=likelihood.num_classes, init_lengthscale=init_lengthscale)
    return(model, likelihood)

### train
def train_DGP_model(train_x, model, likelihood, num_iter=500, lr=0.1, report_iter=50):
    
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
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        if i % report_iter == 0:
            if model.likelihood.second_noise_covar.noise == None:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
                    i + 1, num_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.mean().item()
                ))
            else:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, num_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.mean().item(),
                    model.likelihood.second_noise_covar.noise.mean().item()
                ))
        optimizer.step()
        
    return(model, likelihood)

### eval
# Evaluate DGP
def evaluate_DGP(model, likelihood, test_x, test_y, num_samples=1000):
    
    model.eval()
    likelihood.eval()

    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(test_x)

    pred_samples = test_dist.sample(torch.Size((num_samples,)))
    pred_samples_exp = pred_samples.exp()
    #probs
    pred_samples_norm = (pred_samples_exp / pred_samples_exp.sum(-2, keepdim=True))
    pred_probs = pred_samples_norm.mean(0)
    pred_probs = torch.swapaxes(pred_probs, 0, 1)
    
    # evaluate
    acc = classification_utils.get_accuracy(pred_probs, test_y)
    mnll = classification_utils.mean_neg_log_likelihood(pred_probs, test_y)
    ece, conf, accu, bin_sizes = classification_utils.calibration_test(pred_probs, test_y, nbins=10)
    
    return(acc.item(), mnll.item(), ece.item())



######### LM(Beta)+GP ###########
def LM_beta(alpha, beta):
    
    mu = np.log(alpha/beta)
    var = (alpha+beta)/(alpha*beta)
    return(mu, var)

def transform_y_beta_LM(train_y, eps_alpha=0.01, eps_beta=0.01):

    train_alphas = torch.ones_like(train_y) * eps_alpha
    train_alphas[train_y > 0.5] += 1
    train_betas = torch.ones_like(train_y) * eps_beta
    train_betas[train_y < 0.5] += 1

    train_mu_LB, train_var_LB = LM_beta(train_alphas, train_betas)
    return(train_mu_LB, train_var_LB)

# create model
def create_LM_beta_GP_model(train_x, train_y_mu, train_y_var, learn_additional_noise=False, init_lengthscale=1.):
    
    # We will use the simplest form of GP model, exact inference
    class LM_beta_GPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood, init_lengthscale=init_lengthscale):
            super(LM_beta_GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.covar_module.base_kernel.lengthscale = init_lengthscale

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.sqrt(train_y_var), 
                                                                   learn_additional_noise=learn_additional_noise)
    model = LM_beta_GPModel(train_x, train_y_mu, likelihood)
    return(model, likelihood)

# train
def train_LM_beta_GP_model(train_x, train_y_mu, model, likelihood, num_iter=500, lr=0.1, report_iter=50):
    
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
            if model.likelihood.second_noise_covar.noise == None:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
                    i + 1, num_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.mean().item()
                ))
            else:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, num_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.mean().item(),
                    model.likelihood.second_noise_covar.noise.mean().item()
                ))
        optimizer.step()
        
    return(model, likelihood)

# eval
def logistic(x):
    return(1 / (1 + np.exp(-x)))

def evaluate_LM_beta_GP(model, likelihood, test_x, test_y):
    
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    pred_mean = observed_pred.mean
    pred_probs = logistic(pred_mean)
    # make 2D
    pred_probs = torch.cat([1-pred_probs.view(-1,1), pred_probs.view(-1,1)], dim=1)
    
    # evaluate
    acc = classification_utils.get_accuracy(pred_probs, test_y)
    mnll = classification_utils.mean_neg_log_likelihood(pred_probs, test_y)
    ece, conf, accu, bin_sizes = classification_utils.calibration_test(pred_probs, test_y, nbins=10)
    
    return(acc.item(), mnll.item(), ece.item())


######### LM(Dirichlet)+GP ###########
# translate pseudo counts to Gaussians via LM
def Dirichlet_bridge_mu_batch(alpha):
    K = alpha.size(-1)
    return(torch.log(alpha) - 1/K * torch.sum(torch.log(alpha), dim=1).view(-1, 1))

def Dirichlet_bridge_Sigma_diag_batch(alpha):
    K = alpha.size(-1)
    Sigma_diag = 1/alpha * (1 + 2/K) + 1/K**2 * torch.sum(1/alpha, dim=0)            
    return(Sigma_diag)

def transform_y_Dir_LM(y_train_induced, a_eps=0.00001, num_classes=10):
    train_y_one_hot = torch.nn.functional.one_hot(y_train_induced, num_classes=num_classes) 
    alphas = train_y_one_hot + a_eps

    train_mu_LM = Dirichlet_bridge_mu_batch(alphas)
    train_var_LM = Dirichlet_bridge_Sigma_diag_batch(alphas)
    return(train_mu_LM, train_var_LM)

def create_LM_Dir_GP_model(train_x, train_y_mu, train_y_var, num_classes, init_lengthscale=1, rank=1):
    
    class MultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, num_classes, rank=rank):
            super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_classes
            )
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=num_classes, rank=rank
            )
            self.covar_module.base_kernel.lengthscale = init_lengthscale

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_classes)
    model = MultitaskGPModel(train_x, train_y_mu, likelihood, num_classes=num_classes)
    return(model, likelihood)

def train_LM_Dir_GP_model(train_x, train_y_mu, model, likelihood, num_iter=500, lr=0.1, report_iter=50):
    
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
                model.covar_module.data_covar_module.lengthscale.mean().item(),
                model.likelihood.noise.mean().item()
            ))
        optimizer.step()
        
    return(model, likelihood)

def evaluate_LM_Dir_GP(model, likelihood, test_x, test_y, num_samples=1000):
    
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(test_x)

    pred_samples = test_dist.sample(torch.Size((num_samples,)))
    pred_samples_exp = pred_samples.exp()
    #probs
    pred_samples_norm = (pred_samples_exp / pred_samples_exp.sum(-1, keepdim=True))
    pred_probs = pred_samples_norm.mean(0)
    
    # evaluate
    acc = classification_utils.get_accuracy(pred_probs, test_y)
    mnll = classification_utils.mean_neg_log_likelihood(pred_probs, test_y)
    ece, conf, accu, bin_sizes = classification_utils.calibration_test(pred_probs, test_y, nbins=10)
    
    return(acc.item(), mnll.item(), ece.item())


########## OPTIMIZE INIT LENGTHSCALE ######

# find good initial hyperparameters with k-fold cross validation on the training dataset
def select_init_lengthscale_with_CV(train_x, train_y, mode="DGP", num_inducing_points=200, 
                                    learn_noise=True, num_iter=500, lr=0.1,
                                    lengthscales=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]):
    
    assert mode in ["DGP", "LMGP", "LMGP_con"]
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(train_x):
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        
    # run the k-means clustering
    if mode in ["DGP", "LMGP"]:
        X_train_induced, y_train_induced = classification_utils.k_means_inducing_points(X_train, y_train, num_inducing_points)
    elif mode == "LMGP_con":
        X_train_induced, y_train_induced_alphas, y_train_induced_betas = classification_utils.k_means_inducing_points_LM(X_train, y_train, num_inducing_points)
    
    accuracy_results = []
    nll_results = []
    ece_results = []
    for l in lengthscales:
        
        print("current lengthscale: {}".format(l))
        
        # train the model on inducing points
        if mode == "DGP":
            DGP_model, DGP_likelihood = create_DGP_model(X_train_induced, y_train_induced, init_lengthscale=l, 
                                                         learn_additional_noise=learn_noise)
            DGP_model, DGP_likelihood = train_DGP_model(X_train_induced, DGP_model, DGP_likelihood, num_iter=num_iter, lr=lr, report_iter=num_iter//10)
            acc, mnll, ece = evaluate_DGP(DGP_model, DGP_likelihood, X_test, y_test)
            
        elif mode == "LMGP":
            y_train_induced_mu, y_train_induced_var = transform_y_beta_LM(y_train_induced)
            LMGP_model, LMGP_likelihood = create_LM_beta_GP_model(X_train_induced, y_train_induced_mu,
                                                y_train_induced_var, learn_additional_noise=learn_noise,
                                               init_lengthscale=l)
            LMGP_model, LMGP_likelihood = train_LM_beta_GP_model(X_train_induced, y_train_induced_mu,
                                            LMGP_model, LMGP_likelihood, num_iter=num_iter, lr=lr, report_iter=num_iter//10)
            acc, mnll, ece = evaluate_LM_beta_GP(LMGP_model, LMGP_likelihood, X_test, y_test)
            
        elif mode == "LMGP_con":
            y_train_induced_mu, y_train_induced_var = LM_beta(y_train_induced_alphas, y_train_induced_betas)
            LMGP_model, LMGP_likelihood = create_LM_beta_GP_model(X_train_induced, y_train_induced_mu,
                                                y_train_induced_var, learn_additional_noise=learn_noise,
                                               init_lengthscale=l)
            LMGP_model, LMGP_likelihood = train_LM_beta_GP_model(X_train_induced, y_train_induced_mu,
                                            LMGP_model, LMGP_likelihood, num_iter=num_iter, lr=lr, report_iter=num_iter//10)
            acc, mnll, ece = evaluate_LM_beta_GP(LMGP_model, LMGP_likelihood, X_test, y_test)
        
        # print current results:
        print("acc: {};\t mnll:  {}; \t ece: {}".format(acc, mnll, ece))
        accuracy_results.append(acc)
        nll_results.append(mnll)
        ece_results.append(ece)
        
    res = {
        "lengthscales":lengthscales,
        "accuracy_results":accuracy_results, 
        "nll_results":nll_results,
        "ece_results":ece_results
    }
    
    return(res)

# find good initial hyperparameters with k-fold cross validation on the training dataset (Dirichlet). 
def select_init_lengthscale_with_CV_Dir(train_x, train_y, mode="DGP", num_inducing_points=200, 
                                    learn_noise=True, num_iter=500, lr=0.1,
                                    lengthscales=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
                                    max_test_size=1000, num_classes=10):
    
    assert mode in ["DGP", "LMGP_dir", "LMGP_dir_con"]

    #outer loop
    accuracy_results = []
    nll_results = []
    ece_results = []

    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(train_x):
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        X_test, y_test = X_test[:max_test_size], y_test[:max_test_size]
        
        # run the k-means clustering
        if mode in ["DGP", "LMGP_dir"]:
            X_train_induced, y_train_induced = classification_utils.k_means_inducing_points(X_train, y_train, num_inducing_points)
        elif mode == "LMGP_dir_con":
            X_train_induced, y_train_induced_alphas = k_means_inducing_points_LM_Dir(X_train, y_train, 
                                                                    num_inducing_points, num_classes)
        #classification_utils.
        
        accuracy_results_inner = []
        nll_results_inner = []
        ece_results_inner = []
        for l in lengthscales:
            
            print("current lengthscale: {}".format(l))
            
            # train the model on inducing points
            if mode == "DGP":
                DGP_model, DGP_likelihood = create_DGP_model(X_train_induced, y_train_induced, init_lengthscale=l, 
                                                             learn_additional_noise=learn_noise)
                DGP_model, DGP_likelihood = train_DGP_model(X_train_induced, DGP_model, DGP_likelihood, num_iter=num_iter, lr=lr, report_iter=num_iter//10)
                acc, mnll, ece = evaluate_DGP(DGP_model, DGP_likelihood, X_test, y_test)
                
            elif mode == "LMGP_dir":
                y_train_induced_mu, y_train_induced_var = transform_y_Dir_LM(y_train_induced, num_classes=num_classes)
                LMGP_model, LMGP_likelihood = create_LM_Dir_GP_model(X_train_induced, y_train_induced_mu,
                                                    y_train_induced_var, num_classes=num_classes, rank=num_classes,
                                                   init_lengthscale=l)
                LMGP_model, LMGP_likelihood = train_LM_Dir_GP_model(X_train_induced, y_train_induced_mu,
                                                LMGP_model, LMGP_likelihood, num_iter=num_iter, lr=lr, report_iter=num_iter//10)
                acc, mnll, ece = evaluate_LM_Dir_GP(LMGP_model, LMGP_likelihood, X_test, y_test)
                
            elif mode == "LMGP_dir_con":
    #            y_train_induced_mu, y_train_induced_var = LM_Dir(y_train_induced_alphas, y_train_induced_Dirs)
                y_train_induced_mu = Dirichlet_bridge_mu_batch(y_train_induced_alphas)
                y_train_induced_var = Dirichlet_bridge_Sigma_diag_batch(y_train_induced_alphas)
                LMGP_model, LMGP_likelihood = create_LM_Dir_GP_model(X_train_induced, y_train_induced_mu,
                                                    y_train_induced_var, num_classes=num_classes, rank=num_classes,
                                                   init_lengthscale=l)
                LMGP_model, LMGP_likelihood = train_LM_Dir_GP_model(X_train_induced, y_train_induced_mu,
                                                LMGP_model, LMGP_likelihood, num_iter=num_iter, lr=lr, report_iter=num_iter//10)
                acc, mnll, ece = evaluate_LM_Dir_GP(LMGP_model, LMGP_likelihood, X_test, y_test)
            
            # print current results:
            print("acc: {};\t mnll:  {}; \t ece: {}".format(acc, mnll, ece))
            accuracy_results.append(acc)
            nll_results.append(mnll)
            ece_results.append(ece)

        accuracy_results.append(accuracy_results_inner)
        nll_results.append(nll_results_inner)
        ece_results.append(ece_results_inner)
        
    accuracy_results = np.mean(accuracy_results, 0)
    nll_results = np.mean(nll_results, 0)
    ece_results = np.mean(ece_results, 0)
        
    res = {
        "lengthscales":lengthscales,
        "accuracy_results":accuracy_results, 
        "nll_results":nll_results,
        "ece_results":ece_results
    }
    
    return(res)