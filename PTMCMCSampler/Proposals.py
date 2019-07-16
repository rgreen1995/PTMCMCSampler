#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as ss
import os
import sys
import time


class ProposalError(Exception):
    """Class to handle ProposalErrors

    Parameters
    ----------
    message: str
        the printed error message
    """
    def __init__(self, message):
        super(ProposalError, self).__init__(message)


class JumpProposal(object):
    """Base class for jump proposals

    Parameters
    ----------
    
    Attributes
    ----------
    __name__: str
        name of the class
    """
    def __init__(self, iter):
        self.name = "JumpProposalBaseClass"
        self.iter = iter

    @property
    def __name__(self):
        return self.name

    def return_new_samples(self, samples):
        """Return the new samples

        Parameters
        ----------
        samples: list
            list of new samples
        """
        self.iter += 1
        if self.__name__ == "JumpProposalBaseClass":
            raise ProposalError(
                "JumpProposal is a base class and does not return any samples")
        return samples, 0.0


class SingleComponentAdaptiveCovariance(JumpProposal):
    """Chooses one parameter at a time along covariance
    """
    def __init__(self):
        super(SingleComponentAdaptiveCovariance, self).__init__()
        self.name = "SingleComponentAdaptiveCovariance"

    def __call__(self):
        """
        """
        return super(SingleComponentAdaptiveCovariance, self).__call__()


class AdaptiveCovariance(JumpProposal):
    """Moves in more than one parameter
    """
    def __init__(self):
        super(AdaptiveCovariance, self).__init__()
        self.name = "AdaptiveCovariance"


class SingleComponentAdaptiveGaussian(JumpProposal):
    """Single component adaptive gaussian
    """
    def __init__(self):
        super(SingleComponentAdaptiveGaussian, self).__init__()
        self.name = "SingleComponentAdaptiveGaussian"

    def __call__(self):
        """
        """
        return super(SingleComponentAdaptiveGaussian, self).__call__()


class MultiComponentAdaptiveGaussian(JumpProposal):
    """
    """
    def __init__(self):
        super(MultiComponentAdaptiveGaussian, self).__init__()
        self.name = "MultiComponentAdaptiveGaussian"

    def __call__(self):
        """
        """
        return super(MultiComponentAdpativeGaussian, self).__call__()


class AdaptiveGaussian(JumpProposal):
    """
    """
    def __init__(self):
        super(AdaptiveGaussian, self).__init__()
        self.name = "AdaptiveGaussian"

    def __call__(self):
        """
        """
        return super(AdaptiveGaussian, self).__call__()


class DifferentialEvolution(JumpProposal):
    """Class to handle the differential evolution jump proposal. This jump
    proposal differentially evolves the old sample based on some Gaussian
    randomisation calculated from two existing coordinates.

    Parameters
    ----------

    Attributes
    ----------
    """
    def __init__(self):
        super(DifferentialEvolution, self).__init__()
        self.name = "DifferentialEvolution"

    def __call__(self):
        """
        ""
        return super(DifferentialEvolution, self).__call__()


class Normal(JumpProposal):
    """Class to handle the normal jump proposal. The new sample is drawn from
    a normal distribution centered around the old sample

    Parameters
    ----------

    Attributes
    ----------
    """
    def __init__(self, step_size):
        super(Normal, self).__init__()
        self.name = "Normal"
        self.step_size = step_size

    def __call__(self, samples):
        new_samples = [np.random.normal(i, self.step_size) for i in samples]
        return self.return_new_samples(samples)


class Uniform(JumpProposal):
    """Class to handle a uniform jump proposal. The new sample is drawn from
    a uniform distribution between 2 values

    Parameters
    ----------
    pmin: float
        minimum value of the uniform distribution
    pmax: float
        maximum value fo the uniform distribution
    """
    def __init__(self, pmin, pmax):
        super(Uniform, self).__init__()
        self.pmin = pmin
        self.pmax = pmax

    def __call__(self, samples):
        new_samples = np.random.uniform(self.pmin, self.pmax, len(samples))
        return self.return_new_samples(new_samples)


class Prior(JumpProposal):
    """Draw a proposal from the prior distributions
    """
    def __init__(self):
        super(Prior, self).__init__()

    def __call__(self, sample, prior):
        new_samples = self._draw_from_prior(sample, prior)
        return self.return_new_samples(new_samples)

    def _draw_from_prior(sample, prior):
        """
        """










def covarianceJumpProposalSCAM(x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
    """
    Single Component Adaptive Jump Proposal. This function will occasionally
    jump in more than 1 parameter. It will also occasionally use different
    jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain
    @param groups: index of parameters in which to perform correlated jumps
    @param U : part of covariance matrix svd (need to understand this )
    @param S:  part of covariance matrix svd (need to understand this )
    @param naccepted: the number of accepted samples thus far
    @param chain: The chain history
    @param DEbuffer: Not sure what this is ???
    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability

    """

    q = x.copy()
    qxy = 0

    # choose group
    jumpind = np.random.randint(0, len(groups))
    ndim = len(groups[jumpind])

    # adjust step size
    prob = np.random.rand()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # small-medium jump
    # elif prob > 0.6:
    #:wq    scale = 0.5

    # standard medium jump
    else:
        scale = 1.0

    # scale = np.random.uniform(0.5, 10)

    # adjust scale based on temperature
    if 1 / beta <= 100:
        scale *= np.sqrt(1 / beta)

    # get parmeters in new diagonalized basis

    # make correlated componentwise adaptive jump
    ind = np.unique(np.random.randint(0, ndim, 1))
    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    q[groups[jumpind]] += (
        np.random.randn() * cd * np.sqrt(S[jumpind][ind]) * U[jumpind][:, ind].flatten()
    )

    return q, qxy


# AM jump
def covarianceJumpProposalAM(x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
    """
    Single Component Adaptive Jump Proposal. This function will occasionally
    jump in more than 1 parameter. It will also occasionally use different
    jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain
    @param groups: index of parameters in which to perform correlated jumps
    @param U : part of covariance matrix svd (need to understand this )
    @param S:  part of covariance matrix svd (need to understand this )
    @param naccepted: the number of accepted samples thus far
    @param chain: The chain history
    @param DEbuffer: Not sure what this is ???
    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability

    """

    q = x.copy()
    qxy = 0

    # choose group
    jumpind = np.random.randint(0, len(groups))
    ndim = len(groups[jumpind])

    # adjust step size
    prob = np.random.rand()

    # large jump
    if prob > 0.97:
        scale = 10

    # small jump
    elif prob > 0.9:
        scale = 0.2

    # small-medium jump
    # elif prob > 0.6:
    #    scale = 0.5

    # standard medium jump
    else:
        scale = 1.0

    # adjust scale based on temperature
    if 1.0 / beta <= 100:
        scale *= np.sqrt(1.0 / beta)

    # get parmeters in new diagonalized basis
    y = np.dot(U[jumpind].T, x[groups[jumpind]])

    # make correlated componentwise adaptive jump
    ind = np.arange(len(groups[jumpind]))
    neff = len(ind)
    cd = 2.4 / np.sqrt(2 * neff) * scale

    y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(S[jumpind][ind])
    q[groups[jumpind]] = np.dot(U[jumpind], y)

    return q, qxy


def SCAGjump(x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
    """
    Single Component Adaptive Jump Proposal. This function will occasionally
    jump in more than 1 parameter. It will also occasionally use different
    jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain
    @param groups: index of parameters in which to perform correlated jumps
    @param U : part of covariance matrix svd (need to understand this )
    @param S:  part of covariance matrix svd (need to understand this )
    @param naccepted: the number of accepted samples thus far
    @param chain: The chain history
    @param DEbuffer: Not sure what this is ???
    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability

    """

    q = x.copy()

    # choose parameter
    jumpind = np.random.randint(0, len(q))
    acc_rate = naccepted / iter
    ### ask about this
    scaling_factor = 1.0 / 100

    if np.std(chain[:, jumpind]) == 0:
        current_sigma = 1
    else:
        current_sigma = np.std(chain[:, jumpind])
    if acc_rate > 0.234:
        sigma = current_sigma + q[jumpind] * scaling_factor * acc_rate
    else:
        sigma = current_sigma - q[jumpind] * scaling_factor * acc_rate
    q[jumpind] = q[jumpind] + np.random.normal(0, sigma)
    lqxy = 0
    return q, lqxy


def MCAGjump(x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
    """
    Single Component Adaptive Jump Proposal. This function will occasionally
    jump in more than 1 parameter. It will also occasionally use different
    jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain
    @param groups: index of parameters in which to perform correlated jumps
    @param U : part of covariance matrix svd (need to understand this )
    @param S:  part of covariance matrix svd (need to understand this )
    @param naccepted: the number of accepted samples thus far
    @param chain: The chain history
    @param DEbuffer: Not sure what this is ???
    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability

    """

    q = x.copy()

    # choose parameter
    n_jump_ind = np.random.randint(0, len(q))
    jumpind = np.random.randint(0, len(q), n_jump_ind)
    acc_rate = naccepted / iter
    ### ask about this
    scaling_factor = 1.0 / 100
    for ind in jumpind:
        if np.std(chain[:, ind]) == 0:
            current_sigma = 1
        else:
            current_sigma = np.std(chain[:, ind])
        if acc_rate > 0.234:
            sigma = current_sigma + q[ind] * scaling_factor * acc_rate
        else:
            sigma = current_sigma - q[ind] * scaling_factor * acc_rate
        q[ind] = q[ind] + np.random.normal(0, abs(sigma))
    lqxy = 0
    return q, lqxy


def AGjump(x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
    """
    Single Component Adaptive Jump Proposal. This function will occasionally
    jump in more than 1 parameter. It will also occasionally use different
    jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain
    @param groups: index of parameters in which to perform correlated jumps
    @param U : part of covariance matrix svd (need to understand this )
    @param S:  part of covariance matrix svd (need to understand this )
    @param naccepted: the number of accepted samples thus far
    @param chain: The chain history
    @param DEbuffer: Not sure what this is ???
    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability

    """

    q = x.copy()

    # choose parameter
    acc_rate = naccepted / iter
    ### ask about this
    scaling_factor = 1.0 / 100
    for ind in range(len(q)):
        if np.std(chain[:, ind]) == 0:
            current_sigma = 1
        else:
            current_sigma = np.std(chain[:, ind])
        if acc_rate > 0.234:
            sigma = current_sigma + q[ind] * scaling_factor * acc_rate
        else:
            sigma = current_sigma - q[ind] * scaling_factor * acc_rate
        print(sigma)
        q[ind] = q[ind] + np.random.normal(0, sigma)
    lqxy = 0
    return q, lqxy


# Differential evolution jump
def DEJump(x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
    """
    Single Component Adaptive Jump Proposal. This function will occasionally
    jump in more than 1 parameter. It will also occasionally use different
    jump sizes to ensure proper mixing.

    @param x: Parameter vector at current position
    @param iter: Iteration of sampler
    @param beta: Inverse temperature of chain
    @param groups: index of parameters in which to perform correlated jumps
    @param U : part of covariance matrix svd (need to understand this )
    @param S:  part of covariance matrix svd (need to understand this )
    @param naccepted: the number of accepted samples thus far
    @param chain: The chain history
    @param DEbuffer: Not sure what this is ???
    @return: q: New position in parameter space
    @return: qxy: Forward-Backward jump probability

    """

    # get old parameters
    q = x.copy()
    qxy = 0

    # choose group
    jumpind = np.random.randint(0, len(groups))
    ndim = len(groups[jumpind])

    bufsize = np.alen(DEbuffer)

    # draw a random integer from 0 - iter
    mm = np.random.randint(0, bufsize)
    nn = np.random.randint(0, bufsize)

    # make sure mm and nn are not the same iteration
    while mm == nn:
        nn = np.random.randint(0, bufsize)

    # get jump scale size
    prob = np.random.rand()

    # mode jump
    if prob > 0.5:
        scale = 1.0

    else:
        scale = np.random.rand() * 2.4 / np.sqrt(2 * ndim) * np.sqrt(1 / beta)

    for ii in range(ndim):

        # jump size
        sigma = DEbuffer[mm, groups[jumpind][ii]] - DEbuffer[nn, groups[jumpind][ii]]

        # jump
        q[groups[jumpind][ii]] += scale * sigma

    return q, qxy
