#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .base import JumpProposal


class SingleComponentAdaptiveCovariance(JumpProposal):
    """Chooses one parameter at a time along covariance
    """
    def __init__(self):
        super(SingleComponentAdaptiveCovariance, self).__init__()
        self.name = "SingleComponentAdaptiveCovariance"
        self.check_kwargs(kwargs, self.required_kwargs[self.name])
        self.assign_kwargs(self.required_kwargs[self.name], kwargs)

    def __call__(self, samples):
        return super(SingleComponentAdaptiveCovariance, self).__call__(
            self.jump, samples)

    def jump(self, samples):
        """Return the new samples assuming a Differential Evolution jump
        proposal

        Parameters
        ----------
        samples: list
            list of samples
        """
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

        return new_samples


class AdaptiveCovariance(JumpProposal):
    """Moves in more than one parameter
    """
    def __init__(self):
        super(AdaptiveCovariance, self).__init__()
        self.name = "AdaptiveCovariance"

    def __call__(self, samples):
        super(AdaptiveCovariance, self).__call__(self.jump, samples)

    def jump(self, samples):
        """Return the new samples assuming a Differential Evolution jump
        proposal

        Parameters
        ----------
        samples: list
            list of samples
        """
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

        y = np.dot(self.U[jumpind].T, x[self.groups[jumpind]])

        ind = np.arange(len(self.groups[jumpind]))
        neff = len(ind)
        cd = 2.4 / np.sqrt(2 * neff) * scale

        y[ind] += np.random.randn(neff) * cd * np.sqrt(self.S[jumpind][ind])
        new_samples[self.groups[jumpind]] = np.dot(self.U[jumpind], y)

        return new_samples
