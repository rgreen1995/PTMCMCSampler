#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "SingleComponentAdaptiveCovariance", "AdaptiveCovariance",
    "SingleComponentAdaptiveGaussian", "MultiComponentAdaptiveGaussian",
    "AdaptiveGaussian", "DifferentialEvolution", "Normal", "Uniform",
    "Prior"]


def available_jump_proposals():
    """Print the available jump proposals
    """
    print("Available jump proposals:\n\n%s" % ("\n".join(__all__)))


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
    """
    def __init__(self, iter=None):
        self.iter = iter
        self.name = "JumpProposal"
        self.required_kwargs = {
            "SingleComponentAdaptiveCovariance": ["groups", "beta", "U", "S"],
            "AdaptiveCovariance": ["groups", "beta", "U", "S"],
            "SingleComponentAdaptiveGaussian": ["naccepted", "iter", "chain"],
            "MultiComponentAdaptiveGaussian": ["naccepted", "iter", "chain"],
            "AdaptiveGaussian": ["naccepted", "iter", "chain"],
            "DifferentialEvolution": ["beta", "groups", "DEBuffer"],
            "Normal": ["step_size"],
            "Uniform": ["pmin", "pmax"],
            "Prior": []}

    @property
    def __name__(self):
        return self.name

    def __call__(self, func, samples):
        """
        """
        new_samples = func(samples)
        return self.return_new_samples(new_samples)

    def check_kwargs(self, kwargs, keys):
        """Check that the kwargs are correct for each jump proposal

        Parameters
        ----------
        kwargs: dict
            dictionary of kwargs
        keys: list
            list of kwargs that are needed for a specific jump proposal
        """
        if not all(i in kwargs.keys() for i in keys):
            raise ProposalError(
                "The jump proposal %s requires you to pass the arguments %s. "
                "You have passed %s" % (
                        self.name, " and ".join(keys),
                        " and ".join(kwargs.keys())))

    def assign_kwargs(self, keys, kwargs):
        """Assign the kwargs to the class

        Parameters
        ----------
        kwargs: dict
            dictionary of kwargs
        """
        for i in keys:
            setattr(self, i, kwargs[i])

    def return_new_samples(self, samples):
        """Return the new samples

        Parameters
        ----------
        samples: list
            list of new samples
        """
        if self.iter:
            self.iter += 1
        return samples, 0.0
