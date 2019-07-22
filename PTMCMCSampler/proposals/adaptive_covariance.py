#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .base import JumpProposal


class SingleComponentAdaptiveCovariance(JumpProposal):
    """Chooses one parameter at a time along covariance
    """
    def __init__(self, kwargs=None):
        super(SingleComponentAdaptiveCovariance, self).__init__()
        self.name = "SingleComponentAdaptiveCovariance"

    def __call__(self, samples, kwargs):
        return super(SingleComponentAdaptiveCovariance, self).__call__(
            self.jump, samples, kwargs)

    def jump(self, samples, kwargs):
        """Return the new samples assuming a Differential Evolution jump
        proposal

        Parameters
        ----------
        samples: list
            list of samples
        kwargs: dict
            dictionary of kwargs
        """
        self.check_kwargs(kwargs, self.required_kwargs[self.name])
        self.assign_kwargs(self.required_kwargs[self.name], kwargs)

        new_samples = samples.copy()

        jumpind = np.random.randint(0, len(self.groups))
        ndim = len(self.groups[jumpind])

        prob = np.random.rand()
        if prob > 0.97:
            scale = 10
        elif prob > 0.9:
            scale = 0.2
        else:
            scale = 1.0

        if 1 / self.beta <= 100:
            scale *= np.sqrt(1 / self.beta)

        ind = np.unique(np.random.randint(0, ndim, 1))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        new_samples[self.groups[jumpind]] += (
            np.random.randn() * cd * np.sqrt(self.S[jumpind][ind]) *
            self.U[jumpind][:, ind].flatten()
        )

        return new_samples, 0.0


class AdaptiveCovariance(JumpProposal):
    """Moves in more than one parameter
    """
    def __init__(self, kwargs):
        super(AdaptiveCovariance, self).__init__()
        self.name = "AdaptiveCovariance"

    def __call__(self, samples, kwargs):
        return super(AdaptiveCovariance, self).__call__(
            self.jump, samples, kwargs)

    def jump(self, samples, kwargs):
        """Return the new samples assuming a Differential Evolution jump
        proposal

        Parameters
        ----------
        samples: list
            list of samples
        """
        self.check_kwargs(kwargs, self.required_kwargs[self.name])
        self.assign_kwargs(self.required_kwargs[self.name], kwargs)

        new_samples = samples.copy()

        jumpind = np.random.randint(0, len(self.groups))
        ndim = len(self.groups[jumpind])

        prob = np.random.rand()
        if prob > 0.97:
            scale = 10
        elif prob > 0.9:
            scale = 0.2
        else:
            scale = 1.0

        if 1.0 / self.beta <= 100:
            scale *= np.sqrt(1.0 / self.beta)

        y = np.dot(self.U[jumpind].T, samples[self.groups[jumpind]])

        ind = np.arange(len(self.groups[jumpind]))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        y[ind] += np.random.randn(neff) * cd * np.sqrt(self.S[jumpind][ind])
        new_samples[self.groups[jumpind]] = np.dot(self.U[jumpind], y)

        return new_samples, 0.0
