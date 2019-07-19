==============
PTMCMC Samples
==============

Jump Proposals
-----------------

This implementation uses an adaptive jump proposal scheme by default using both standard and single component Adaptive Metropolis (AM) and Differential Evolution (DE) jumps.

Default Jump Proposals
-------------------------

:code:`PTMCMC` already has a list of available jump proposals implemented. Each jump proposal will have an equal weighting. This means that each jump proposal has equal probability of being picked.

SingleComponentAdaptiveCovariance
##################################
This jump chooses one parameter at a time along covariance. This function will occasionally jump in more than 1 parameter. It will also occasionally use different jump sizes to ensure proper mixing.

AdaptiveCovariance
##################
This jump moves in more than one parameter. This function will occasionally jump in more than 1 parameter. It will also occasionally use different jump sizes to ensure proper mixing.

SingleComponentAdaptiveGaussian
################################
This jump chooses one parameter at random and moves according to a normal distribution where sigma is an adaptive parameter that adjusts according to the current acceptance rate.

MultiComponentAdaptiveGaussian
###############################
This jump will pick several parameters at random and moves according to a normal distribution where sigma is an adaptive parameter that adjusts according to the current acceptance rate.

AdaptiveGaussian
##################
This jump will pick all parameters and move them according to a normal distribution where sigma is an adaptive parameter that adjusts according to the current acceptance.

DifferentialEvolution
#####################
This jump proposal differentially evolves the old sample based on some Gaussian randomisation calculated from two existing coordinates.

Normal
########
The new sample is drawn from a normal distribution centred around the old sample.

Uniform
########
The new sample is drawn from a uniform distribution between 2 values

Prior
######

Weighting Default Jump Proposals
---------------------------------

You are able to specify different weights for different jump proposals by passing a dictionary called weights to the sampler object. This dictionary might have the following form:

.. code-block:: python

   weights = {"SingleComponentAdaptiveCovariance": 5,
           "AdaptiveCovariance": 1,
           "SingleComponentAdaptiveGaussian": 1,
           "MultiComponentAdaptiveGaussian": 1,
           "AdaptiveGaussian": 1,
           "DifferentialEvolution": 1}

Custom Jump Proposals
--------------------------------

Custom jump proposals can be added very easily. Lets say that you would like to add the following jump proposal.
