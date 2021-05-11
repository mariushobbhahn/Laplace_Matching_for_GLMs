# Laplace_Matching_for_GLMs

This is the official github repository for the paper "Laplace Matching for fast Approximate Inference in Generalized Linear Models" by me, Marius Hobbhahn, and Philipp Hennig. The paper can be found on arxiv at: https://arxiv.org/abs/2105.03109

Everything that can be found in the paper should be reproducible by running the experiments in this repository. 

The figures are not uploaded as files because of their site. You can create them by running the respective jupyter notebook files.

## Derivations Bridges

This folder contains
- The implementation for all individual bridges, i.e. the forward and backward transformation to the latent base for exponential, Gamma, inverse Gamma, Chi-square, Beta, Dirichlet, Wishart, and inverse-Wishart distribution. 
- The computation of the KL divergences.

## Poisson Experiment

This folder contains the simple experiment, i.e. the number of awards based on the math score. It can be interpreted as a proof of concept for Laplace Matching. 

## Dirichlet Experiment

Votes in an election are categorical data. The Dirichlet distribution is a conjugate prior to categoricals. Thus we can create a Dirichlet distribution for the likelihood in Bayesian inference. We use Laplace Matching to create a latent Gaussian Process from these Dirichlets and use the softmax transformation to transform posterior GP-samples back to probability space. We use this rather complicated setting because it shows both the simplicity and the power of Laplace Matching.

This folder contains
- The Experiment on the German national elections. From this we generate the overall election plots over time from 1949 to 2017. It also contains an ESS implementation for comparison.
- The Experiment on the Tübingen local elections. From this we generate the beta marginal plots and the accuracy vs. number of samples plot for mean and std estimates. It also contains an ESS implementation for comparison.
- An implementation of multiclass GP-classification. This is used as a comparison, for the timing and in general since this might be one of the choices that people would want to use for election scenarios. 

## Wishart Experiment

Covariance matrices are symmetric positive-definite (spd) matrices. The conjugate prior for spd matrices is the (inverse-)Wishart distribution. We transform the Wishart likelihood to a Gaussian with Laplace Matching. We then update a latent Gaussian Process with these Wishart distributions and transform posterior GP-samples back to the correct base through the respective transformation (i.e. XX.T or expm(X)). 

## Final Words

I'm always looking for feedback. If you have problems understanding the content don't hesitate to contact me. 
If you have any further questions or suggestions about this repo you can write me at marius.hobbhahn[at]gmail.com
