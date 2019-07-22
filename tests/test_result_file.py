import pytest
import numpy as np
from PTMCMCSampler.result import Result


class BaseResult(object):
    """Base class for Result object
    """
    def test_samples(self):
        """Test that the samples property returns what it should
        """
        result_values = self.result.samples.all()
        assert result_values == self.initial_samples[:, self.burnin :].all()

    def test_likelihood_value(self):
        """Test that the likelihood property returns what it should
        """
        result_values = self.result.likelihood_values.all()
        assert result_values == self.initial_likelihood_vals[:, self.burnin :].all()

    def test_prior_value(self):
        """Test that the likelihood property returns what it should
        """
        result_values = self.result.prior_values.all()
        assert result_values == self.initial_prior_vals[:, self.burnin :].all()

    def test_set_burnin(self):
        """Test the function `set_burnin`
        """
        assert self.result.burnin == self.burnin
        self.result.set_burnin(500)
        assert self.result.burnin == 500
        self.result.set_burnin(self.burnin)
        assert self.result.burnin == self.burnin


class TestResult1d(BaseResult):
    """Class to test the Result object for 1 cold chains
    """
    def setup(self):
        """Setup the Result object
        """
        self.n_cold_chains = 1
        self.initial_samples = np.array([np.array(
            [np.random.random(2) for i in range(1000)])] * self.n_cold_chains)
        self.initial_likelihood_vals = np.array([np.array(
            [np.random.random(1) for i in range(1000)])] * self.n_cold_chains)
        self.initial_prior_vals = np.array([np.array(
            [np.random.random(1) for i in range(1000)])] * self.n_cold_chains)

        self.burnin = 100
        self.result = Result(
            self.initial_samples, burnin=self.burnin,
            initial_likelihood_vals=self.initial_likelihood_vals,
            initial_prior_vals=self.initial_prior_vals,
            num_chains=self.n_cold_chains
            )

    def test_set_burnin(self):
        """Test the function `set_burnin`
        """
        super(TestResult1d, self).test_set_burnin()

    def test_samples(self):
        """Test that the samples property returns what it should
        """
        super(TestResult1d, self).test_samples()

    def test_likelihood_value(self):
        """Test that the likelihood property returns what it should
        """
        super(TestResult1d, self).test_likelihood_value()

    def test_prior_value(self):
        """Test that the likelihood property returns what it should
        """
        super(TestResult1d, self).test_prior_value()


class TestResult2d(BaseResult):
    """Class to test the Result object for 2 cold chains
    """
    def setup(self):
        """Setup the Result object
        """
        self.n_cold_chains = 2
        self.initial_samples = np.array([np.array(
            [np.random.random(2) for i in range(1000)])] * self.n_cold_chains)
        self.initial_likelihood_vals = np.array([np.array(
            [np.random.random(1) for i in range(1000)])] * self.n_cold_chains)
        self.initial_prior_vals = np.array([np.array(
            [np.random.random(1) for i in range(1000)])] * self.n_cold_chains)

        self.burnin = 100
        self.result = Result(
            self.initial_samples, burnin=self.burnin,
            initial_likelihood_vals=self.initial_likelihood_vals,
            initial_prior_vals=self.initial_prior_vals,
            num_chains=self.n_cold_chains
            )

    def test_set_burnin(self):
        """Test the function `set_burnin`
        """
        super(TestResult2d, self).test_set_burnin()

    def test_samples(self):
        """Test that the samples property returns what it should
        """
        super(TestResult2d, self).test_samples()

    def test_likelihood_value(self):
        """Test that the likelihood property returns what it should
        """
        super(TestResult2d, self).test_likelihood_value()

    def test_prior_value(self):
        """Test that the likelihood property returns what it should
        """
        super(TestResult2d, self).test_prior_value()
