#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .base import JumpProposal


class Normal(JumpProposal):
    """Class to handle the normal jump proposal. The new sample is drawn from
    a normal distribution centered around the old sample

    Parameters
    ----------
    kwargs: dict
        dictionary of kwargs
    """
    def __init__(self, kwargs):
        super(Normal, self).__init__()
        self.name = "Normal"
        self.check_kwargs(kwargs, self.required_kwargs[self.name])
        self.assign_kwargs(self.required_kwargs[self.name], kwargs)

    def __call__(self, samples, kwargs):
        return super(Normal, self).__call__(self.jump, samples, kwargs)

    def jump(self, samples, kwargs):
        """Return the new samples assuming a Normal jump proposal

        Parameters
        ----------
        samples: list
            list of samples
        """
        new_samples = [np.random.normal(i, self.step_size) for i in samples]
        return new_samples, 0.0
