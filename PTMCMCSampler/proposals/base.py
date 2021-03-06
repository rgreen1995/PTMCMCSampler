#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "SingleComponentAdaptiveCovariance", "AdaptiveCovariance",
    "SingleComponentAdaptiveGaussian", "MultiComponentAdaptiveGaussian",
    "AdaptiveGaussian", "DifferentialEvolution", "Normal", "Uniform",
    "MALA", "HMC", "Prior"]

__default__ = [
    "SingleComponentAdaptiveCovariance", "AdaptiveCovariance",
    "SingleComponentAdaptiveGaussian", "MultiComponentAdaptiveGaussian",
    "AdaptiveGaussian", "DifferentialEvolution"
    ]


def available_jump_proposals():
    """Print the available jump proposals
    """
    print("Available jump proposals:\n\n%s" % ("\n".join(__all__)))


def default_jump_proposals():
    """Print the default jump proposals
    """
    print("Default jump proposals:\n\n%s" % ("\n".join(__default__)))


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
            "Gradient": ["loglik_grad", "logprior_grad", "mm_inv", "nburn"],
            "HMC": ["step_size", "nminsteps", "nmaxsteps"],
            "Prior": []}

    @property
    def __name__(self):
        return self.name

    def __call__(self, func, samples, kwargs):
        new_samples, forward_backward_prob = func(samples, kwargs)
        return self.return_new_samples(new_samples, forward_backward_prob)

    def check_kwargs(self, kwargs, keys):
        """Check that the kwargs are correct for each jump proposal

        Parameters
        ----------
        kwargs: dict
            dictionary of kwargs
        keys: list
            list of kwargs that are needed for a specific jump proposal
        """
        if kwargs == None and keys != []:
            raise ProposalError(
                "The jump proposal %s requires you to pass the arguments %s. "
                "Please pass the arguments with the "
                "`initialize_jump_proposal_arguments` kwarg to the sampler "
                "object" % (self.name, " and ".join(keys)))

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

    def return_new_samples(self, samples, forward_backward_prob):
        """Return the new samples

        Parameters
        ----------
        samples: list
            list of new samples
        """
        if self.iter:
            self.iter += 1
        return samples, forward_backward_prob
