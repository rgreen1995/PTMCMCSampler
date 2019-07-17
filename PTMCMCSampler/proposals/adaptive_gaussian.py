#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .base import JumpProposal


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
        self.check_kwargs(kwargs, self.required_kwargs[self.name])
        self.assign_kwargs(self.required_kwargs[self.name], kwargs)

    def __call__(self):
        return super(MultiComponentAdpativeGaussian, self).__call__(
            self.jump, self.samples)

    def jump(self, samples):
        """Return the new samples assuming a Differential Evolution jump
        proposal

        Parameters
        ----------
        samples: list
            list of samples
        """
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
    """
    """
    def __init__(self):
        super(AdaptiveGaussian, self).__init__()
        self.name = "AdaptiveGaussian"
        self.check_kwargs(kwargs, self.required_kwargs[self.name])
        self.assign_kwargs(self.required_kwargs[self.name], kwargs)

    def __call__(self):
        return super(AdaptiveGaussian, self).__call__(self.jump, self.samples)

    def jump(self, samples):
        """Return the new samples assuming a Differential Evolution jump
        proposal

        Parameters
        ----------
        samples: list
            list of samples
        """
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
            new_samples += np.random.normal(0, sigma)
        return new_samples
