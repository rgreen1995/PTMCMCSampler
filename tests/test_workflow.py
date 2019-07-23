import pytest
import numpy as np
from PTMCMCSampler.result import Result
from PTMCMCSampler import PTMCMCSampler
import shutil


class GaussianLikelihood(object):
    """Class used to define the Guassian Likelihood
    """
    def __init__(self, ndim=2, pmin=-10, pmax=10):
        self.a = np.ones(ndim)*pmin
        self.b = np.ones(ndim)*pmax

    def lnlikefn(self, x):
        return -0.5*np.sum(x**2)-len(x)*0.5*np.log(2*np.pi)

    def lnlikefn_grad(self, x):
        ll = -0.5*np.sum(x**2)-len(x)*0.5*np.log(2*np.pi)
        ll_grad = -x
        return ll, ll_grad

    def lnpriorfn(self, x):
        if np.all(self.a <= x) and np.all(self.b >= x):
            return 0.0
        else:
            return -np.inf
        return 0.0

    def lnpriorfn_grad(self, x):
        return self.lnpriorfn(x), np.zeros_like(x)

    def lnpost_grad(self, x):
        ll, ll_grad = self.lnlikefn_grad(x)
        lp, lp_grad = self.lnpriorfn_grad(x)
        return ll+lp, ll_grad+lp_grad

    def lnpost(self, x):
        return lnpost_grad(x)[0]


class TestWorkflow(object):
    """Test the workflow from start to end and make sure that there are no
    failures
    """
    def setup(self):
        """Setup everything to run the workflow
        """
        self.ndim = 2
        self.pmin, self.pmax = 0.0, 10.0
        self.glo = GaussianLikelihood(
            ndim=self.ndim, pmin=self.pmin, pmax=self.pmax)
        self.p0 = np.random.uniform(self.pmin, self.pmax, self.ndim)
        self.cov = np.eye(self.ndim) * 0.1**2
        self.sampler = PTMCMCSampler.PTSampler(
            self.ndim, self.glo.lnlikefn, self.glo.lnpriorfn, np.copy(self.cov),
            logl_grad=self.glo.lnlikefn_grad, logp_grad=self.glo.lnpriorfn_grad,
            outDir='./test_chains')

    def teardown(self):
        """Remove the files created from PTMCMCSampler
        """
        shutil.rmtree("./test_chains")

    def test_sample(self):
        """Try running the workflow with all default jump proposals
        """
        data = self.sampler.sample(
            self.p0, 5000, burn=500, thin=1, covUpdate=500)
        assert isinstance(data, Result)
