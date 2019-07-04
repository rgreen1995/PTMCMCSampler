#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as ss
import os
import sys
import time


def covarianceJumpProposalSCAM(
    self, x, iter, beta, groups, U, S, naccepted, chain, DEbuffer
):
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
def covarianceJumpProposalAM(
    self, x, iter, beta, groups, U, S, naccepted, chain, DEbuffer
):
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


def SCAGjump(self, x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
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


def MCAGjump(self, x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
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


def AGjump(self, x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
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
def DEJump(self, x, iter, beta, groups, U, S, naccepted, chain, DEbuffer):
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
