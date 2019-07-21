#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .base import JumpProposal


class SingleComponentAdaptiveGaussian(JumpProposal):
    """Class to handle a single component Adaptive Gaussian jump. This jump
    chooses one parameter at random and moves according to a normal
    distribution where sigma is an adaptive parameter that adjusts according
    to the current acceptance rate
    """
    def __init__(self, kwargs=None):
        super(SingleComponentAdaptiveGaussian, self).__init__()
        self.name = "SingleComponentAdaptiveGaussian"

    def __call__(self, samples, kwargs):
        return super(SingleComponentAdaptiveGaussian, self).__call__(
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

        # choose parameter
        jumpind = np.random.randint(0, len(new_samples))
        acc_rate = self.naccepted / self.iter

        scaling_factor = 1./100

        if np.std(self.chain[:, jumpind]) == 0 :
            current_sigma = 1
        else :
            current_sigma = np.std(self.chain[:, jumpind])

        scaled_samples = new_samples[jumpind] * scaling_factor * acc_rate

        if acc_rate > 0.234:
            sigma = current_sigma + scaled_samples
        else:
            sigma = current_sigma - scaled_samples

        new_samples[jumpind] += np.random.normal(0, sigma)
        return new_samples


class MultiComponentAdaptiveGaussian(JumpProposal):
    """Class to handle a multiple component Adaptive Gaussian jump. This jump
    will pick several parameters at random and moves according to a normal
    distribution where sigma is an adaptive parameter that adjusts according
    to the current acceptance rate

    Parameters
    ----------
    kwargs: dict
        dictionary of kwargs
    """
    def __init__(self, kwargs=None):
        super(MultiComponentAdaptiveGaussian, self).__init__()
        self.name = "MultiComponentAdaptiveGaussian"

    def __call__(self, samples, kwargs):
        return super(MultiComponentAdaptiveGaussian, self).__call__(
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
        n_jump_ind = np.random.randint(0, len(new_samples))
        jumpind = np.random.randint(0, len(new_samples), n_jump_ind)
        acc_rate = self.naccepted / self.iter

        scaling_factor = 1.0 / 100
        for ind in jumpind:
            if np.std(self.chain[:, ind]) == 0:
                current_sigma = 1
            else:
                current_sigma = np.std(self.chain[:, ind])

            scaled_samples = new_samples[ind] * scaling_factor * acc_rate
            if acc_rate > 0.234:
                sigma = current_sigma + scaled_samples
            else:
                sigma = current_sigma - scaled_samples
            new_samples[ind] += np.random.normal(0, abs(sigma))
        return new_samples


class AdaptiveGaussian(JumpProposal):
    """Class to handle an Adaptive Gaussian jump. This jump will pick all
    parameters and move them according to a normal distribution where sigma is
    an adaptive parameter that adjusts according to the current acceptance
    rate
    """
    def __init__(self, kwargs=None):
        super(AdaptiveGaussian, self).__init__()
        self.name = "AdaptiveGaussian"

    def __call__(self, samples, kwargs):
        return super(AdaptiveGaussian, self).__call__(self.jump,
            samples, kwargs)

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

        acc_rate = self.naccepted / self.iter
        scaling_factor = 1.0 / 100

        for ind in range(len(new_samples)):
            if np.std(self.chain[:, ind]) == 0:
                current_sigma = 1
            else:
                current_sigma = np.std(self.chain[:, ind])

            scaled_samples = new_samples[ind] * scaling_factor * acc_rate
            if acc_rate > 0.234:
                sigma = current_sigma + scaled_samples
            else:
                sigma = current_sigma - scaled_samples
            new_samples[ind] += np.random.normal(0, sigma)
        return new_samples
