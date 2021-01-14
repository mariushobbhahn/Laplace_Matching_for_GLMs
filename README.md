# Laplace_Matching_for_GLMs
Github repo for the paper. Not yet finished. Will link if ready. 

Everything that can be found in the paper should be reproducible by running the experiments in this repository. 

## Derivations_Bridges

This folder contains
- The implementation for all individual bridges, i.e. the forward and backward transformation to the latent base
- The computation of the KL divergences. Sometimes they are done through their pdfs and sometimes through samples.

## Poisson_Experiment

This folder contains the simple experiment, i.e. the number of awards based on the math score. It can be interpreted as a proof of concept.

## Dirichlet_Experiment

Votes in an election are categorical data. The Dirichlet distribution is a conjugate prior to categoricals. We use Laplace Matching to create a latent Gaussian Process from these Dirichlets and use the softmax transformation to transform GP-samples back to probability space. We use this rather complex setting because it shows both the simplicity and the power of Laplace Matching.

This folder contains
- The Experiment on the German national elections. From this we generate the overall election plots over time from 1949 to 2017. It also contains an ESS implementation for comparison.
- The Experiment on the Tübingen local elections. From this we generate the beta marginal plots and the accuracy vs. number of samples plot for mean and std estimates. It also contains an ESS implementation for comparison.
- An implementation of multiclass GP-classification. This is used as a comparison and for the timing and in general since this might be one of the choices that people would want to use for election scenarios. 

## Wishart_Experiment

Covariance matrices are symmetric positive-definite matrices. The conjugate prior for psd matrices is the (inverse) Wishart distribution. We use Laplace Matching to create a latent Gaussian Process from these Wishart distributions and transform GP-samples back to the correct base through the respective transformation (i.e. X@X.T). 

## Final Words

I'm always looking for feedback. If you have problems understanding a section or research ideas don't hesitate to contact me. 
If you have any further questions or suggestions about this repo you can contact me at marius.hobbhahn[at]gmail.com
