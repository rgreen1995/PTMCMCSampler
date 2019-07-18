#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .base import JumpProposal


class DifferentialEvolution(JumpProposal):
    """Class to handle the differential evolution jump proposal. This jump
    proposal differentially evolves the old sample based on some Gaussian
    randomisation calculated from two existing coordinates.

    Parameters
    ----------
    kwargs: dict
        dictionary of kwargs
    """
    def __init__(self, kwargs):
        super(DifferentialEvolution, self).__init__()
        self.name = "DifferentialEvolution"

    def __call__(self, samples, kwargs):
        return super(DifferentialEvolution, self).__call__(self.jump, self.samples, kwargs)

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
        ndim = len(groups[jumpind])

        bufsize = np.alen(self.DEbuffer)

        mm = np.random.randint(0, bufsize)
        nn = np.random.randint(0, bufsize)
        while mm == nn:
            nn = np.random.randint(0, bufsize)

        prob = np.random.rand()
        if prob > 0.5:
            scale = 1.0
        else:
            rand = np.random.rand()
            scale = rand * 2.4 / np.sqrt(2 * ndim) * np.sqrt(1 / self.beta)

        for ii in range(ndim):
            first_term = self.DEbuffer[mm, groups[jumpind][ii]]
            second_term = self.DEbuffer[nn, groups[jumpind][ii]]
            sigma = first_term - second_term
            new_samples[groups[jumpind][ii]] += scale * sigma
        return new_samples

        
