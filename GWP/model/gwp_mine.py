import sys
import numpy as np
import emcee
import matplotlib.pyplot as plt

# calculate GP stuff
from numpy.random import randn, multivariate_normal
from numpy.linalg import cholesky, solve
from scipy.linalg import cho_solve, cho_factor
import time
from scipy.stats import multivariate_normal as mv_norm
from .kernels_mine import sum_kernel
import torch

####### Elliptical slice sampling ########

#
# Elliptical slice sampling; taken from a github repo; 
# credits belong to original author
#
import math
def elliptical_slice(initial_theta,prior,lnpdf,pdf_params=(),
                     cur_lnpdf=None,angle_range=None):
    """
    NAME:
       elliptical_slice
    PURPOSE:
       Markov chain update for a distribution with a Gaussian "prior" factored out
    INPUT:
       initial_theta - initial vector
       prior - cholesky decomposition of the covariance matrix 
               (like what numpy.linalg.cholesky returns), 
               or a sample from the prior
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params= parameters to pass to the pdf
       cur_lnpdf= value of lnpdf at initial_theta (optional)
       angle_range= Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
    OUTPUT:
       new_theta, new_lnpdf
    HISTORY:
       Originally written in matlab by Iain Murray (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
       2012-02-24 - Written - Bovy (IAS)
    """
    D= len(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf= lnpdf(initial_theta,*pdf_params)

    # Set up the ellipse and the slice threshold
    if len(prior.shape) == 1: #prior = prior sample
        nu= prior
        print("first condition is True")
    else: #prior = cholesky decomp
        if not prior.shape[0] == D or not prior.shape[1] == D:
            raise IOError("Prior must be given by a D-element sample or DxD chol(Sigma)")
        nu= np.dot(prior,np.random.normal(size=D))
    hh = math.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi= np.random.uniform()*2.*math.pi
        phi_min= phi-2.*math.pi
        phi_max= phi
    else:
        # Randomly center bracket on current point
        phi_min= -angle_range*np.random.uniform()
        phi_max= phi_min + angle_range
        phi= np.random.uniform()*(phi_max-phi_min)+phi_min

        
    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        xx_prop = (initial_theta*math.cos(phi)).reshape(-1) + (nu*math.sin(phi)).reshape(-1)
        cur_lnpdf = lnpdf(xx_prop,*pdf_params)
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
        # Propose new angle difference
        phi = np.random.uniform()*(phi_max - phi_min) + phi_min
    return (xx_prop,cur_lnpdf)

def is_pos_def(x):
    eig = np.linalg.eigvals(x)
    pd = np.all(eig > 0)
    if not pd:
        raise(ValueError("matrix is not positive definite! Matrix: {}; Eigenvalues: {}".format(x, eig)))
    else:
        return(True)
    
def create_spd_matrix(p=2, eps=10e-3):
    
    #create two symmetric positive-definite matrices
    X_ = torch.rand(p,p)
    X_sym = (X_ + X_.T)/2
    lambda_X_min = torch.abs(torch.min(torch.eig(X_sym, False)[0])) + eps
    X_spd = X_sym + torch.eye(p) * lambda_X_min
    #print("X_spd is spd: ", is_pos_def(X_spd))
    
    return(X_spd.cpu().numpy())


"""
Contains the code for the generalized Wishart process model as described by
Wilson and Ghahramani in their paper https://arxiv.org/abs/1101.0240.
"""

class GeneralizedWishartProcess(object):
    """
    Fits and predicts a GWP model.
    """
    def __init__(self, sig_var, kernel, tau_prior_mean, tau_prior_var,
                 L_prior_var, use_sum_kernel=False, kernel2=None, sig_var2=1, 
                 tau_prior_mean2=1, tau_prior_var2=0.1, add_ind_mean=False):
        """
        Initialize a generalized Wishart process model with the given
        parameters.

        sig_var: The signal variance for the kernel.
        kernel:  The kernel function to use. Must accept arguments like
                 (t1, t2, tau).
        tau_prior_mean: The mean of the LogNormal prior over tau.
        tau_prior_var: The variance of the LogNormal prior over tau.
        L_prior_var: The prior (Gaussian) variance over the elements of L.
        """
        self.parameters = {
            'SIG_VAR': sig_var,
            'KERNEL': kernel,
            'TAU_PRIOR_MEAN': tau_prior_mean,
            'TAU_PRIOR_VAR': tau_prior_var,
            'L_PRIOR_VAR': L_prior_var,
            'USE_SUM_KERNEL':False,
            'ADD_IND_MEAN':add_ind_mean
        }
        
        if use_sum_kernel:
            self.parameters['USE_SUM_KERNEL'] = True
            self.parameters['KERNEL2'] = kernel2
            self.parameters['SIG_VAR2'] = sig_var2
            self.parameters['TAU_PRIOR_MEAN2'] = tau_prior_mean2
            self.parameters['TAU_PRIOR_VAR2'] = tau_prior_var2

    def _construct_kernel(self, taus, times, plot=False):
        """
        Construct the kernel for the GPs which generate the GWP.

        params: The array of tau parameters, of size Nu * N. #can also be one dimensional
        times: The times to compute the pairwise kernel function at.
        
        Returns a νNT x νNT kernel matrix.
        """

        n_kernel = self.Nu_plus * self.D * self.T
        K = np.eye(n_kernel)
        if self.parameters['USE_SUM_KERNEL']:
            k1 = self.parameters['KERNEL']
            k2 = self.parameters['KERNEL2']
            tau1 = taus[0]
            tau2 = taus[1]
            sig1 = self.parameters['SIG_VAR']
            sig2 = self.parameters['SIG_VAR2']
            #print("taus: ", taus)
        else:
            k = self.parameters['KERNEL']
        
        """
        for nu in range(self.Nu):
            for n in range(self.D): #N is number of dimensions
                for t1 in range(self.T):
                    for t2 in range(t1, self.T): #this made it asymmetric ... 
                        if t1 != t2: #make sure that diagonal is 1 as demanded in paper
                            i = t1 * self.Nu * self.D + self.D * nu + n
                            j = t2 * self.Nu * self.D + self.D * nu + n
                            p = nu * self.D + n
                            if self.parameters['USE_SUM_KERNEL']:
                                k_value = sum_kernel(times[t1], times[t2], k1, k2, [tau1[p], sig1], [tau2[p], sig2])
                            else:
                                k_value = k(times[t1], times[t2], [tau, self.parameters['SIG_VAR']])
                            #k_value = k(times[t1], times[t2], [tau, self.parameters['SIG_VAR']])
                            K[i, j] = k_value
                            K[j, i] = k_value
        #"""
        
        for nu in range(self.Nu_plus):
            for d in range(self.D): #N is number of dimensions
                for n1 in range(self.T):
                    for n2 in range(n1, self.T): 
                        if n1 != n2: #make sure that diagonal is 1 as demanded in paper
                            p = (nu * self.D +  d)
                            i = p * self.T + n1
                            j = p * self.T + n2
                            if self.parameters['USE_SUM_KERNEL']:
                                k_value = sum_kernel(times[n1], times[n2], k1, k2, [sig1, tau1[p], 16], [sig2, tau2[p]])
                            else:
                                k_value = k(times[n1], times[n2], [self.parameters['SIG_VAR'], taus[p], 16])
                                #k_value = k(times[n1], times[n2], [self.parameters['SIG_VAR'], 1, taus[p]]) #adapt periodicity
                            #k_value = k(times[t1], times[t2], [tau, self.parameters['SIG_VAR']])
                            K[i, j] = k_value
                            K[j, i] = k_value
                               
        #print("psd: ", is_pos_def(K))
        
        #add small nugget on diagonal to prevent numerical instabilities
        #K += 1e-8 * np.eye(np.shape(K)[0])
        
        if plot:
            plt.figure(figsize=(15,15))
            plt.imshow(K)
            plt.colorbar()
            plt.show();
        
        return(K)

    def compute_sigma(self, L, u):
        """
        Compute the covariance matrix for a specific timepoint.

        L: The lower cholesky decomposition of the scale parameter for the
           Wishart distribution (of dimension N x N).
        u: The fitted GP function values that generate the draw from the Wishart
           distribution, of size Nu * N.
        We assume u to be given in the form [u_11, u_12, ..., u_nu,d]

        Returns the D x D covariance matrix.
        """
        Sig = np.zeros_like(L)
        for nu in range(self.Nu):
            idx_ = nu * self.D
            Sig += np.outer((L @ u[idx_:(idx_+self.D)]), (u[idx_:(idx_+self.D)] @ L.T))

        return Sig

    def _log_data_likelihood(self, u, L):
        """
        Compute the likelihood of observing the data given the model parameters
        u and L which completely determine Sigma. We use the simplest possible
        data likelihood: sum over all times t in 1, ..., T and computing the
        log-probability of observing the data given that it comes from the
        distribution r(t) ~ N(0, Sigma(t)).

        u:    The vector of GP function values, of size Nu * N * T.
        L:    The lower Cholesky factor of the scale Wishart prior.

        Returns the log-likelihood of observing the data given the model
        parameters.
        """
        loglik = 0
        """
        for t in range(T):
            idx = self.Nu * self.D * t
            #TODO: change this to cholesky
            #Siginv = np.linalg.inv(
            #    self.compute_sigma(L, u[idx:(idx + self.Nu*self.D)])
            #)
            #term = -0.5*np.matmul(self.data[:, t].T, np.matmul(Siginv, self.data[:, t]))
            L_chol = cho_factor(self.compute_sigma(L, u[idx:(idx + self.Nu*self.N)]))
            mu = self._mean_function1(u[idx:(idx + self.Nu*self.N)], L)
            term = -0.5*(self.data[:, t]-mu).T @ cho_solve(L_chol, (self.data[:, t] - mu).T).T
            loglik += term
        """
        
        u_ = u.reshape(self.Nu_plus * self.D, self.T) #reshape such that cols are one u vector
        """
        plt.figure(figsize=(10,10))
        plt.imshow(u_)
        plt.show();
        """
        for t in range(self.T):
            u_current = u_[:self.D*self.Nu, t] #go through col by col
            assert(len(u_current) == self.Nu * self.D)
            Sig_ = self.compute_sigma(L, u_current)
            L_chol = cho_factor(Sig_)
            #mu = self._mean_function1(u_current, L)
            if self.parameters['ADD_IND_MEAN']:
                #mu = self._mean_function3(u_[t])
                mu = self._mean_function3(u_[:, t])
            else:
                mu = np.zeros(self.D)
            assert(len(mu) == self.D)
            norm_const_ = -self.D/2 * np.log(2 * np.pi) - 1/2 * np.log(np.linalg.det(Sig_))
            term = norm_const_ - 0.5*(self.data[:, t]-mu).T @ cho_solve(L_chol, (self.data[:, t] - mu).T).T
            loglik += term
            
        return loglik
    
    def _mean_function1(self, u, L):
        """
        mean function 1 as presented in the original paper in equation 24
        We assume u to be given in the form [u_11, u_12, ..., u_nu,d]
        """
        
        mu1 = np.zeros(self.D)
        for nu in range(self.Nu):
            idx_ = nu * self.D
            mu1 += L @ u[idx_:(idx_+self.D)]

        return(mu1)
    
    def _mean_function3(self, u):
        """
        mean function 3 as presented in the original paper in equation 26
        We assume u to be given in the form [u_11, u_12, ..., u_nu,d]
        """
        return(u[-self.D:])
        
    
    def _mean_function2(self, u, L):
        """
        mean function 2 as presented in the original paper in equation 25
        We assume u to be given in the form [u_11, u_12, ..., u_nu,d]
        """
        mu2 = np.zeros(self.D)
        for nu in range(self.Nu):
            idx_ = nu * self.D
            mu1 += L @ u[idx_:(idx_+self.D)]
        
        mu2 += u[-self.D:]
        return(mu2)
    
    def _invert_block_kernel(self, K_B, mult):
        """
        We have a block diagonal Kernel K_B. It has Nu*D blocks of sice N.
        Instead of inverting the entire Kernel, we can also invert all blocks independently.
        Since we use the Cholesky decomposition for inversion we can directly multiply the kernel 
        with a vector or matrix to save time, e.g. in x.T @ Sigma @ x
        """
        res = np.zeros_like(mult)
        for block in range(self.Nu_plus * self.D):
            idx_ = block * self.T
            K_block = K_B[idx_:(idx_+self.T), idx_:(idx_+self.T)]
            mult_block = mult[idx_:(idx_+self.T)]
            L_chol = cho_factor(K_block)
            mult_solve = cho_solve(L_chol, mult_block.T).T
            res[idx_:(idx_+self.T)] = mult_solve
            
        return(res)

    def _sample_u(self, f, tau, L):
        """
        Sample the vector of GP function values given a previous setting of the
        parameter. We use elliptical slice sampling, specifically a direct
        implementation of the algorithm in figure 2 from the original ESS paper.

        f: The previous setting of u.
        tau: The current tau parameters. 
        L: The current L parameter.

        Returns the newly sampled u and its posterior probability as a tuple.
        """
        Nu = self.Nu
        data = self.data
        T = self.T

        if self.parameters['USE_SUM_KERNEL']:
            tau = tau.reshape(2, -1)
        K = self._construct_kernel(tau, range(T))
        #use cholesky to invert the Kernel
        #""" #use an alternate implementation of ESS
        L_chol, _ = cho_factor(K, lower=True)
        def log_likelihood_function(f, L=L):
            #return(-0.5*f @ self._invert_block_kernel(K_B=K, mult=f).T)
            return(self._log_data_likelihood(f, L))
        f_x, ll = elliptical_slice(initial_theta=f, prior=L_chol, lnpdf=log_likelihood_function) 
        return(f_x, ll)
        #"""
        """
        #ellipse = np.random.multivariate_normal(np.zeros(K.shape[0]), K) #using a zero mean function
        ellipse = mv_norm.rvs(np.zeros(K.shape[0]), K) #using a zero mean function
        u = np.random.uniform()
        logy = self._log_data_likelihood(f, L) + np.log(u)
        angle = np.random.uniform(high=2*np.pi)
        angle_min, angle_max = angle - 2*np.pi, angle
        while True:
            fp = f*np.cos(angle) + ellipse*np.sin(angle)
            log_data_lik = self._log_data_likelihood(fp, L)
            if log_data_lik > logy:
                #log_u_lik = -0.5*np.matmul(fp, np.matmul(Kinv, fp))
                #log_u_lik = -0.5*fp @ cho_solve(L_chol, fp.T).T
                log_u_lik = -0.5*fp @ self._invert_block_kernel(K_B=K, mult=fp).T
                return fp, log_data_lik + log_u_lik
            else:
                if angle < 0:
                    angle_min = angle
                else:
                    angle_max = angle
                angle = np.random.uniform(angle_min, angle_max)
         """
                
    def _sample_logtau(self, logtau, u, L, MCMC=True, verbose=False, MC_samples=1):
        """
        Sample the next log(tau) parameter given a previous setting. We use the
        standard Metropolis-Hastings implementation in `emcee` to get the next
        sample.

        logtau: The previous setting of the log(tau) parameters. #can also be just one log-tau
        u: The current setting of u.
        L: The current setting of L.

        Returns a tuple consisting of the newly sampled log(tau) parameter and
        its posterior probability.
        """
        Nu = self.Nu
        
        def _log_logtau_prob(logtaup):
            T = self.data.shape[1]
            if self.parameters['USE_SUM_KERNEL']:
                logtaup = logtaup.reshape(2, -1)
            K = self._construct_kernel(np.exp(logtaup), range(T))
            #det_prob = -0.5 * np.log(np.linalg.det(K)) #numerical problems because K is so large
            log_u_prob = -0.5*u @ self._invert_block_kernel(K_B=K, mult=u).T
            mean = self.parameters['TAU_PRIOR_MEAN']
            var = self.parameters['TAU_PRIOR_VAR']
            log_prior = np.sum(-0.5*((logtaup - mean)**2/var))
            #log_prior = np.sum(-0.5*((logtaup)**2))

            return(log_u_prob + log_prior)
            #return(log_prior)

        if MCMC:
            #print("drawing logtau MCMC!")
            dim = np.prod(logtau.shape)
            scale = 1/1000 #dim
            sampler = emcee.MHSampler(np.eye(dim) * scale, dim=dim, lnprobfn=_log_logtau_prob)
            #the scale parameter seems to help prevent the MC sampler to get stuck
            logtaup, _, _ = sampler.run_mcmc(logtau.reshape(-1), MC_samples)
            #acceptance_fraction = sampler.acceptance_fraction
            #autocorr = sampler.acor
            #print("acceptance fraction: ", acceptance_fraction)
            #print("autocorr: ", autocorr)
        else:
            logtaup = logtau
            
        if verbose:
            print("Logtau: {}; {}".format(np.shape(logtaup), logtaup))

        return logtaup, _log_logtau_prob(logtaup)
    
    
    def _sample_L(self, L, u, MCMC=True, verbose=False, MC_samples=1):
        """
        Sample the next L parameter given the previous setting. We use the
        standard Metropolis-Hastings implementation in `emcee` to get the next
        sample.

        L: The previous setting of the L parameter.
        u: The current setting of the u parameter.

        Returns a tuple consisting of the newly sampled L parameter and its
        posterior probability.
        """
        Nu = self.Nu
        data = self.data
        
        def _log_L_prob(Lp):
            Lpm = np.zeros(L.shape)
            Lpm[np.tril_indices(L.shape[0])] = Lp
            log_prior = np.sum(-0.5 * Lp**2 / self.parameters['L_PRIOR_VAR'])

            return(self._log_data_likelihood(u, Lpm) + log_prior)
            #return(log_prior)
        
        if MCMC:
            dim = int((L.shape[0]**2 + L.shape[0])/2)
            scale= 1/10 #dim
            sampler = emcee.MHSampler(np.eye(dim) * scale, dim=dim, lnprobfn=_log_L_prob)
            #the scale denotes the correlation the samples should have
            Lp, _, _ = sampler.run_mcmc(L[np.tril_indices(L.shape[0])], MC_samples)
            Lpm = np.zeros(L.shape)
            Lpm[np.tril_indices(L.shape[0])] = Lp
        else:
            Lp  = L[np.tril_indices(L.shape[0])]
            Lpm = L
            
        if verbose:
            print("L: {}; {}".format(np.shape(Lpm), Lpm)) 
        
        return Lpm, _log_L_prob(Lp)

    def _init_u(self, T, tau):
        """
        Initialize the u parameter.

        T: The number of timepoints for the model fit.
        tau: A random setting of tau.

        Returns a random setting of u.
        """
        if self.parameters['USE_SUM_KERNEL']:
            tau = tau.reshape(2, -1)
        K = self._construct_kernel(tau, range(T))
        draw = np.random.multivariate_normal(np.zeros(K.shape[0]), K)

        return(draw)

    def _init_logtau(self):
        """
        Initialize the log(tau) parameter.

        Returns a random setting of log(tau).
        """
        mean = self.parameters['TAU_PRIOR_MEAN']
        var = self.parameters['TAU_PRIOR_VAR']
        #"""
        if self.parameters['USE_SUM_KERNEL']:
            mean2 = self.parameters['TAU_PRIOR_MEAN2']
            var2 = self.parameters['TAU_PRIOR_VAR2']
            tau1 = np.random.normal(mean, var, size=self.Nu_plus * self.D)
            tau2 = np.random.normal(mean2, var2, size=self.Nu_plus * self.D)
            return(np.stack([tau1, tau2]))
        else:
            return(np.random.normal(mean, var, size=self.Nu_plus * self.D))
        #"""
        #one-dimensional case, i.e. one tau for all kernels
        #return(np.random.normal(mean, var, size=1))

    def _init_L(self, N, data=None, use_data=False):
        """
        Initialize the L parameter to the identity.

        Returns the identity matrix scaled by the prior variance of L.
        """
        D = self.D #dimensions
        if use_data:
            nu = self.Nu
            T = self.T #datapoints
            data_centered = data - data.mean(1).reshape(-1, 1)
            emp_cov = 1/T * data_centered @ data_centered.T #because the data is transposed
            
            L, _ = cho_factor(1/nu * emp_cov, lower=True)
            assert(np.shape(L)[0] == D)
            return(L)# * self.parameters['L_PRIOR_VAR'])
        
        else:
            L = np.eye(D)
            return(L * self.parameters['L_PRIOR_VAR'])

    def fit(self, data, init=None, numit=1000, progress=10, sample_tau=True, sample_L=True, add_ind_mean=False):
        """
        Fit the model using a Gibbs sampling routine.

        data: The data to fit on. Dimension N x T, where N is the number of
              assets and T is the number of timepoints. Element (n, t) is the
              return of the nth asset at time t.
        init: A dict containing an initialization for each of the parameters.
              Must include keys 'logtau', 'u', and 'L'.

        Returns the chain of samples and diagnostics (likelihood and
        posterior probabilities).
        """
        samples, diagnostics = [], []
        self.data = data

        D, T = data.shape
        Nu = D + 1
        if add_ind_mean:
            Nu_plus = Nu + 1  #if you want to use mean 2 or mean 3 as given in equation 25 and 26
            self.Nu_plus = Nu_plus
        else:
            Nu_plus = Nu
            self.Nu_plus = Nu_plus
        self.Nu, self.D, self.T = Nu, D, T

        if init:
            logtau = init['logtau']
            u = init['u']
            L = init['L']
        else:
            u = self._init_u(T, np.exp(self._init_logtau()))
            logtau = self._init_logtau()
            L = self._init_L(D, data, use_data=True)

        samples.append([u, np.exp(logtau), L])
        
        #plot the prior kernel once
        self._construct_kernel(np.exp(logtau), times=range(T), plot=True)


        for it in range(numit):
            verbose_ = False
            if progress and it % progress is 0:
                verbose_ = True
            
            data_lik = self._log_data_likelihood(u, L)
            u, u_prob = self._sample_u(u, np.exp(logtau), L)
            logtau, logtau_prob = self._sample_logtau(logtau, u, L, MCMC=sample_tau, verbose=verbose_, MC_samples=1)
            L, L_prob = self._sample_L(L, u, MCMC=sample_L, verbose=verbose_, MC_samples=1)
            
            samples.append([u, np.exp(logtau), L])
            diagnostics.append([data_lik, u_prob, logtau_prob, L_prob])

            if verbose_:
                print(
                    "Iter {}: loglik = {:.2f}, log P(u|...) = {:.2f}, log P(tau|...) = {:.2f}, log P(L|...) = {:.2f}".format(it, *diagnostics[-1])
                )
                sys.stdout.flush()
                
        #plot the final kernel once
        if self.parameters['USE_SUM_KERNEL']:
            logtau = logtau.reshape(2, -1)
        self._construct_kernel(np.exp(logtau), times=range(T), plot=True)

        self.samples = samples
        self.diagnostics = np.asarray(diagnostics)
        print("Optimal likelihood: {:.3f}".format(np.max(self.diagnostics[:, 0])))

        return samples, diagnostics

    def optimal_params(self, burnin=200):
        """
        Return the maximum a posteriori setting of parameters from the model.

        burnin: The number of iterations to consider burnin. These iterations
                will be ignored when computing the MAP estimate.

        Returns the optimal sample, which is a list containing the optimal
        settings of u, tau, and L in that order.
        """
        return self.samples[np.argmax(self.diagnostics[burnin:, 0]) + burnin]

    def _predict_next_u(self, T, burnin=200):
        """
        Predict u for the next timepoint from the model.

        T: The number of timepoints used to train the model. Timepoints
           0, ..., T - 1 were used to train the model, and timepoint T will be
           predicted.
        burnin: The number of iterations to consider burnin. These iterations
                will be ignored when computing the MAP estimate of the
                parameters used for the prediction.

        Returns the MAP estimate of the next timepoint's u.
        """
        """
        u, tau, L = self.optimal_params(burnin)

        K = self._construct_kernel(tau, range(T + 1))
        idxs = np.full(K.shape[0], True)                     #array of size (N) filled with True booleans
        idxs[np.asarray(list(range(T, int(len(u)/T*(T+1)), T + 1)))] = False  # start the range at T, stop at n/V * V+1, have T+1 steps; I feel like there is a +T+1 missing n the stop part.
        Kbinv = np.linalg.inv(K[idxs, :][:, idxs])            # see equation 20 in the og paper
        A = K[np.logical_not(idxs), :][:, idxs]               # take the kernel at all other indeces
        ustar = np.matmul(np.matmul(A, Kbinv), u)             # prediction of the new u
        
        return ustar
        """
        pass

    def predict_next_timepoint(self, data, burnin=200):
        """
        Predict the covariance at the next timepoint from the model using the
        MAP estimate of the parameters.

        data: The data used to fit the model.
        burnin: The number of iterations to be considered burn-in. These
                iterations will be ignored when computing the MAP estimate.

        Returns the MAP estimate of the next timepoint's covariance.
        """
        u, tau, L = self.optimal_params(burnin)
        ustar = self._predict_next_u(data.shape[1], burnin)

        return self.compute_sigma(L, ustar)
    
    def cov_from_samples(self, samples, num_samples=100):
        
        selected_us = np.concatenate(samples[:,0]).reshape(num_samples, -1)
        selected_Ls = np.concatenate(samples[:,2]).reshape(num_samples, -1)
        
        Sigma_outer = []
        
        for i in range(num_samples):
            u = selected_us[i].reshape(-1)
            L = selected_Ls[i].reshape(self.D, self.D)
            
            Sigma_inner = []            
            
            u_ = u.reshape(self.Nu_plus * self.D, self.T) #reshape such that rows are one u vector
            for t in range(self.T):
                u_current = u_[:self.D*self.Nu, t] #go through col by col
                assert(len(u_current) == self.Nu * self.D)
                Sigma_ = self.compute_sigma(L, u_current)
                Sigma_inner.append(Sigma_)
                
            Sigma_outer.append(np.array(Sigma_inner))

        return(np.array(Sigma_outer))

    def draw_train_samples(self, num_samples=100, c=5):
        """
        draw samples at the training points from the posterior after training
        """
        
        selected_samples = np.array(self.samples[-num_samples*c::c])
        cov_samples = self.cov_from_samples(selected_samples, num_samples=num_samples)
        return(np.array(cov_samples))
    
    def draw_samples_prior(self, data, init=None, num_samples=100, verbose=False, add_ind_mean=False):
        """
        Sample from the priors of L and tau and return the resulting covariance matrices

        data: The data to fit on. Dimension N x T, where N is the number of
              assets and T is the number of timepoints. Element (n, t) is the
              return of the nth asset at time t.
        init: A dict containing an initialization for each of the parameters.
              Must include keys 'logtau', 'u', and 'L'.

        Returns the chain of samples and diagnostics (likelihood and
        posterior probabilities).
        """
        samples = []
        self.data = data

        D, T = data.shape
        Nu = D + 1
        if add_ind_mean:
            Nu_plus = Nu + 1  #if you want to use mean 2 or mean 3 as given in equation 25 and 26
            self.Nu_plus = Nu_plus
        else:
            Nu_plus = Nu
            self.Nu_plus = Nu_plus
        self.Nu, self.D, self.T = Nu, D, T
        
        if init:
            logtau = init['logtau']
            u = init['u']
            L = init['L']
        else:
            u_prior = self._init_u(T, np.exp(self._init_logtau()))
            logtau_prior = self._init_logtau()
            L_prior = self._init_L(D, data, use_data=True)

        samples.append([u_prior, np.exp(logtau_prior), L_prior])
        
        #plot the prior kernel once
        self._construct_kernel(np.exp(logtau_prior), times=range(T), plot=True)


        for it in range(num_samples):
            
            u_prior = self._init_u(T, np.exp(self._init_logtau()))
            logtau_prior = self._init_logtau()
            L_prior = self._init_L(D, data, use_data=True)
            
            samples.append([u_prior, logtau_prior, L_prior])
            
        self.samples = samples
        
        cov_samples = self.cov_from_samples(np.array(samples[-num_samples:]), num_samples=num_samples)

        return samples, cov_samples
        
        
        