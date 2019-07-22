import pytest
import numpy as np
from PTMCMCSampler.result import Result


class TestResult(object):
    """Class to test the Result object
    """
    def setup(self):
        """Setup the Result object
        """
        self.initial_samples = np.array(
            [np.random.random(2) for i in range(1000)])
        self.burnin = 100
        self.result = Result(self.initial_samples, burnin=self.burnin)

    def test_set_burnin(self):
        """Test the function `set_burnin`
        """
        assert self.result.burnin == self.burnin
        self.result.set_burnin(500)
        assert self.result.burnin == 500
        self.result.set_burnin(self.burnin)
        assert self.result.burnin == self.burnin

    def test_samples(self):
        """Test that the samples property returns what it should
        """
        self.result.samples.all() == self.initial_samples[self.burnin:].all()

        
