#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .base import JumpProposal


class Uniform(JumpProposal):
    """Class to handle a uniform jump proposal. The new sample is drawn from
    a uniform distribution between 2 values

    Parameters
    ----------
    kwargs: dict
        dictionary of kwargs
    """
    def __init__(self, kwargs):
        super(Uniform, self).__init__()
        self.name = "Uniform"
        self.check_kwargs(kwargs, self.required_kwargs[self.name])
        self.assign_kwargs(self.required_kwargs[self.name], kwargs)

    def __call__(self, samples, kwargs):
        return super(Uniform, self).__call__(self.jump, samples, kwargs)

    def jump(self, samples, kwargs):
        """Return the new samples assuming a Uniform jump proposal

        Parameters
        ----------
        samples: list
            list of samples
        """
        new_samples = np.random.uniform(self.pmin, self.pmax, len(samples))
        return new_samples, 0.0
