import sys
import numpy as np
import emcee

# calculate GP stuff
from numpy.random import randn, multivariate_normal
from numpy.linalg import cholesky, solve
from scipy.linalg import cho_solve, cho_factor
import time


"""
Contains the code for the generalized Wishart process model as described by
Wilson and Ghahramani in their paper https://arxiv.org/abs/1101.0240.

IMPORTANT: the code was created by https://github.com/wjn0/gwp; I merely pulled, updated and improved it a little
"""

class GeneralizedWishartProcess(object):
    """
    Fits and predicts a GWP model.
    """
    def __init__(self, sig_var, kernel, tau_prior_mean, tau_prior_var,
                 L_prior_var):
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
            'MH_L_SCALE': L_prior_var/10,
            'TAU_SCALE': tau_prior_var/100
        }

    def _construct_kernel(self, params, times):
        """
        Construct the kernel for the GPs which generate the GWP.

        params: The array of tau parameters, of size Nu * N.
        times: The times to compute the pairwise kernel function at.
        
        Returns a νNT x νNT kernel matrix.
        """
        def kidx(n, t):
            """
            Generates a lambda which maps indices of u to their flattened
            positions.
            """
            return lambda a, b, c: a * (n * t) + b * n + c
        T = len(times)

        K = np.eye(np.prod([self.Nu, self.N, T]))
        k = self.parameters['KERNEL']
        for nu in range(self.Nu):
            for n in range(self.N):
                for t1 in range(T):
                    for t2 in range(t1, T):
                        if t1 != t2:
                            i = t1 * self.Nu * self.N + self.N * nu + n
                            j = t2 * self.Nu * self.N + self.N * nu + n
                            p = nu * self.N + n
                            K[i, j] = k(times[t1], times[t2], [params[p],
                                                               self.parameters['SIG_VAR']])

        return K

    def compute_sigma(self, L, u):
        """
        Compute the covariance matrix for a specific timepoint.

        L: The lower cholesky decomposition of the scale parameter for the
           Wishart distribution (of dimension N x N).
        u: The fitted GP function values that generate the draw from the Wishart
           distribution, of size Nu * N.

        Returns the N x N covariance matrix.
        """
        Sig = np.zeros(L.shape)
        for nu in range(self.Nu):
            idx = nu * self.N
            Sig += np.matmul(L, np.matmul(
                np.outer(u[idx:(idx+self.N)], u[idx:(idx+self.N)]),
                L.T
            ))

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
        T = self.data.shape[1]
        for t in range(T):
            idx = self.Nu * self.N * t
            Siginv = np.linalg.inv(
                self.compute_sigma(L, u[idx:(idx + self.Nu*self.N)])
            )
            term = -0.5*np.matmul(self.data[:, t].T, np.matmul(Siginv, self.data[:, t]))
            loglik += term

        return loglik

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
        T = self.data.shape[1]

        K = self._construct_kernel(tau, range(T))
        #Kinv = np.linalg.inv(K)
        #use cholesky to invert the Kernel
        L_chol = cho_factor(K)

        ellipse = np.random.multivariate_normal(np.zeros(K.shape[0]), K)
        u = np.random.uniform()
        logy = self._log_data_likelihood(f, L) + np.log(u)
        angle = np.random.uniform(high=2*np.pi)
        angle_min, angle_max = angle - 2*np.pi, angle
        while True:
            fp = f*np.cos(angle) + ellipse*np.sin(angle)
            log_data_lik = self._log_data_likelihood(fp, L)
            if log_data_lik > logy:
                #log_u_lik = -0.5*np.matmul(fp, np.matmul(Kinv, fp))
                log_u_lik = -0.5*fp @ cho_solve(L_chol, fp.T).T
                return fp, log_data_lik + log_u_lik
            else:
                if angle < 0:
                    angle_min = angle
                else:
                    angle_max = angle
                angle = np.random.uniform(angle_min, angle_max)

    def _sample_logtau(self, logtau, u, L):
        """
        Sample the next log(tau) parameter given a previous setting. We use the
        standard Metropolis-Hastings implementation in `emcee` to get the next
        sample.

        logtau: The previous setting of the log(tau) parameters.
        u: The current setting of u.
        L: The current setting of L.

        Returns a tuple consisting of the newly sampled log(tau) parameter and
        its posterior probability.
        """
        Nu = self.Nu
        data = self.data
        T = data.shape[1]

        def log_logtau_prob(logtaup):
            K = self._construct_kernel(np.exp(logtaup), range(T))
            #Kinv = np.linalg.inv(K) #K is psd and thus should be inverted using cholesky decompositions
            L_chol = cho_factor(K)
            #log_u_prob = -0.5*np.matmul(u, np.matmul(Kinv, u))
            log_u_prob = -0.5* u @ cho_solve(L_chol, u.T).T
            mean = self.parameters['TAU_PRIOR_MEAN']
            var = self.parameters['TAU_PRIOR_VAR']
            log_prior = np.sum(-0.5*((logtaup - mean)**2/var))

            return log_u_prob + log_prior

        dim = np.prod(logtau.shape)
        sampler = emcee.MHSampler(np.eye(dim), dim=dim,
                                  lnprobfn=log_logtau_prob)
        logtaup, _, _ = sampler.run_mcmc(logtau, 1)

        return logtaup, log_logtau_prob(logtaup)

    def _sample_L(self, L, u):
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

        def log_L_prob(Lp):
            Lpm = np.zeros(L.shape)
            Lpm[np.tril_indices(L.shape[0])] = Lp
            log_prior = np.sum(-0.5 * Lp**2 / self.parameters['L_PRIOR_VAR'])

            return self._log_data_likelihood(u, Lpm) + log_prior

        dim = int((L.shape[0]**2 + L.shape[0])/2)
        scale = self.parameters['MH_L_SCALE']
        sampler = emcee.MHSampler(np.eye(dim) * scale, dim=dim,
                                  lnprobfn=log_L_prob)
        Lp, _, _ = sampler.run_mcmc(L[np.tril_indices(L.shape[0])], 1)
        Lpm = np.zeros(L.shape)
        Lpm[np.tril_indices(L.shape[0])] = Lp
        
        return Lpm, log_L_prob(Lp)

    def _init_u(self, T, tau):
        """
        Initialize the u parameter.

        T: The number of timepoints for the model fit.
        tau: A random setting of tau.

        Returns a random setting of u.
        """
        K = self._construct_kernel(tau, range(T))
        draw = np.random.multivariate_normal(np.zeros(K.shape[0]), K)

        return draw

    def _init_logtau(self):
        """
        Initialize the log(tau) parameter.

        Returns a random setting of log(tau).
        """
        return np.random.normal(size=self.Nu * self.N)

    def _init_L(self, N):
        """
        Initialize the L parameter to the identity.

        Returns the identity matrix scaled by the prior variance of L.
        """
        L = np.eye(N)

        return L * self.parameters['L_PRIOR_VAR']

    def fit(self, data, init=None, numit=1_000, progress=10):
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

        N, T = data.shape
        Nu = N + 1
        self.Nu, self.N = Nu, N

        if init:
            logtau = init['logtau']
            u = init['u']
            L = init['L']
        else:
            u = self._init_u(T, np.exp(self._init_logtau()))
            logtau = self._init_logtau()
            L = self._init_L(N)

        samples.append([u, np.exp(logtau), L])

        for it in range(numit):
            data_lik = self._log_data_likelihood(u, L)
            u, u_prob = self._sample_u(u, np.exp(logtau), L)
            logtau, logtau_prob = self._sample_logtau(logtau, u, L)
            L, L_prob = self._sample_L(L, u)
            
            samples.append([u, np.exp(logtau), L])
            diagnostics.append([data_lik, u_prob, logtau_prob, L_prob])

            if progress and it % progress is 0:
                print(
                    "Iter {}: loglik = {:.2f}, log P(u|...) = {:.2f}, log P(tau|...) = {:.2f}, log P(L|...) = {:.2f}".format(it, *diagnostics[-1])
                )
                sys.stdout.flush()

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
        u, tau, L = self.optimal_params(burnin)

        K = self._construct_kernel(tau, range(T + 1))
        idxs = np.full(K.shape[0], True)                     #array of size (N) filled with True booleans
        idxs[np.asarray(list(range(T, int(len(u)/T*(T+1)), T + 1)))] = False  # start the range at T, stop at n/V * V+1, have T+1 steps; I feel like there is a +T+1 missing n the stop part.
        Kbinv = np.linalg.inv(K[idxs, :][:, idxs])            # see equation 20 in the og paper
        A = K[np.logical_not(idxs), :][:, idxs]               # take the kernel at all other indeces
        ustar = np.matmul(np.matmul(A, Kbinv), u)             # prediction of the new u
        
        return ustar

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

    def draw_train_samples(self, data, burnin=200):
        """
        Predict u for the training points, i.e. just draw some samples
        
        Function is supposed to be a merger of _predict_next_u and predict_next_timepoint
        """
        u, tau, L = self.optimal_params(burnin)
        N = len(data)
        
        K = self._construct_kernel(tau, range(N))
        
        #TODO: invert K with cholesky decomposition not np.linalg.inv
        #G_ = K_XX + Y_Sigma_block
        #t0 = time.time()
        #G = cho_factor(G_)
        #t1 = time.time()
        #print("time for cho factor: ", t1 - t0)
        
        
        #predict at indices
        
        
