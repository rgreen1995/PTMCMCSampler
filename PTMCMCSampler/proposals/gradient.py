"""
Implementation of the No-U-Turn-Sampler. Code follows algorithm 6 from the NUTS
paper (Hoffman & Gelman, 2011)

reference: arXiv:1111.4246
"The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte
Carlo", Matthew D. Hoffman & Andrew Gelman

Rutger van Haasteren
"""

from .base import JumpProposal
import scipy.linalg as sl
import numpy as np


class Gradient(JumpProposal):
    """Class to hand jumps that use gradient information

    Parameters
    ----------
    kwargs: dict
        dictionary of kwargs to initalize the class
    """
    def __init__(self, kwargs):
        super(Gradient, self).__init__()
        self.name = "Gradient"
        self.check_kwargs(kwargs, self.required_kwargs[self.name])
        self.assign_kwargs(self.required_kwargs[self.name], kwargs)
        self.ndim = len(self.mm_inv)
        self.set_cf()
        self.epsilon = None
        self.beta = 1.0
        self.iter = 0.0

        print("WARNING: GradientJumps not yet adaptive. Choose cov wisely!")

    def set_cf(self):
        """Update the Cholesky factor of the inverse mass matrix
        """
        self.cov_cf = sl.cholesky(self.mm_inv, lower=True)
        self.cov_cfi = sl.solve_triangular(self.cov_cf, np.eye(len(self.cov_cf)), trans=0, lower=True)

    def update_cf(self):
        """Update the Cholesky factor of the inverse mass matrix
        
        NOTE: this function is different from the one in GradientJump!
        """
        new_cov_cf = sl.cholesky(self.mm_inv, lower=True)

        ldet_old = np.sum(np.log(np.diag(self.cov_cf)))
        ldet_new = np.sum(np.log(np.diag(new_cov_cf)))

        self.cov_cf = np.exp((ldet_old-ldet_new)/self.ndim) * new_cov_cf
        self.cov_cfi = sl.solve_triangular(self.cov_cf, np.eye(len(self.cov_cf)), trans=0, lower=True)

    def func_grad(self, samples):
        """Return the Log-prob and gradient corrected for the temperature

        Parameters
        ----------
        samples: numpy.array
            array of samples
        """
        ll, ll_grad = self.loglik_grad(samples)
        lp, lp_grad = self.logprior_grad(samples)
        return self.beta*ll+lp, self.beta*ll_grad+lp_grad

    def forward(self, samples):
        """Return whitened parameters via a coordinate transformation

        Parameters
        ----------
        samples: numpy.array
            array of samples
        """
        return np.dot(self.cov_cfi.T, samples)

    def backward(self, samples):
        """Return un-whitened parameters via a coordinate transformation. Note
        this is the inverse of 'forward'

        Parameters
        ----------
        samples: numpy.array
            array of samples
        """
        return np.dot(self.cov_cf.T, samples)

    def func_grad_white(self, samples):
        """Return a whitened version of func_grad

        Parameters
        ----------
        samples: numpy.array
            array of samples
        """
        x = self.backward(samples)
        fv, fg = self.func_grad(x)
        return fv, np.dot(self.cov_cf, fg)

    def draw_momenta(self):
        """Draw new momentum variables
        """
        return np.random.randn(len(self.mm_inv))

    def loghamiltonian(self, logl, r):
        """Return the value of the Hamiltonian

        Parameters
        ----------
        logl: float
            log likelihood
        r: numpy.array
            position
        """
        try:
            return logl-0.5*np.dot(r, r)
        except ValueError as err:
            return np.nan

    def posmom_inprod(self, theta, r):
        """
        """
        try:
            return np.dot(theta, r)
        except ValueError as err:
            return np.nan

    def leapfrog(self, theta, r, grad, epsilon):
        """Perform a leapfrog jump in the Hamiltonian space

        Parameters
        ----------
        theta: numpy.array
            Initial parameter position
        r: float
            Initial momentum
        grad: numpy.array
            Initial gradient
        epsilon: float
            step size
        """
        rprime = r + 0.5 * epsilon * grad
        thetaprime = theta + epsilon * rprime
        logpprime, gradprime = self.func_grad_white(thetaprime)
        rprime = rprime + 0.5 * epsilon * gradprime

        return thetaprime, rprime, gradprime, logpprime


class MALA(Gradient):
    """Perform a MALA jump
    """
    def __init__(self, kwargs):
        """Initialize the MALA Jump"""
        super(MALA, self).__init__(kwargs)
        self.name = "MALA"
        self.cd = 2.4/np.sqrt(self.ndim)
        self.set_eigvecs()

    def set_eigvecs(self):
        """Set the eignvectors of the mass matrix
        """
        self._u = np.eye(self.ndim)
        self._s = np.ones(self.ndim)

    def __call__(self, samples, kwargs):
        return super(MALA, self).__call__(self.jump, samples, kwargs)

    def jump(self, samples, kwargs):
        """Return the new samples assuming a MALA jump proposal

        Parameters
        ----------
        samples: list
            list of samples
        """
        samples = np.atleast_1d(samples)
        if len(np.shape(samples)) > 1:
            raise ValueError('samples is expected to be a 1-D array')

        new_samples0 = self.forward(samples)
        logp, grad0 = self.func_grad_white(new_samples0)

        # Choose an eigenvector to jump in, and the size
        i = np.random.randint(0, self.ndim)
        vec = self._u[i,:]
        val = self._s[i]
        dist = np.random.randn()

        # Do the leapfrog
        mq0 = new_samples0 + 0.5 * vec * self.cd**2 * np.dot(vec, grad0)/2 / val
        new_samples1 = mq0 + dist * vec * self.cd / np.sqrt(val)
        logp1, grad1 = self.func_grad_white(new_samples1)
        mq1 = new_samples1 + 0.5 * vec * self.cd**2 * np.dot(vec, grad1)/2 / val

        qxy = 0.5 * (np.sum((mq0-new_samples1)**2/val) - \
            np.sum((mq1-new_samples0)**2/val))

        return self.backward(new_samples1), qxy


class HMC(Gradient):
    """Perform a Hamiltonian Monte-Carlo jump
    """
    def __init__(self, kwargs):
        super(HMC, self).__init__(kwargs)
        self.name = "HMC"
        self.check_kwargs(kwargs, self.required_kwargs[self.name])
        self.assign_kwargs(self.required_kwargs[self.name], kwargs)

    def __call__(self, samples, kwargs):
        return super(HMC, self).__call__(self.jump, samples, kwargs)

    def jump(self, samples, kwargs):
        """Return the new samples assuming a HMC jump proposal

        Parameters
        ----------
        samples: list
            list of samples
        """
        samples = np.atleast_1d(samples)
        if len(np.shape(samples)) > 1:
            raise ValueError('samples is expected to be a 1-D array')

        new_samples0 = self.forward(samples)
        qxy = 0
        logp0, grad0 = self.func_grad_white(new_samples0)

        p0 = self.draw_momenta()
        joint0 = self.loghamiltonian(logp0, p0)

        nsteps = np.random.randint(self.nminsteps, self.nmaxsteps)
        p, new_samples, grad = np.copy(p0), np.copy(new_samples0), np.copy(grad0)

        for ii in range(nsteps):
            new_samples1, p1, grad1, logp1 = self.leapfrog(
                new_samples, p, grad, self.step_size)
            joint1 = self.loghamiltonian(logp1, p1)
            p, new_samples = np.copy(p1), np.copy(new_samples1)
            grad = np.copy(grad1)

            if (joint1 - 1000.) < joint0:
                break

        qxy = joint1 - joint0

        return self.backward(new_samples), qxy
