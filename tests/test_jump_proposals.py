import pytest
import numpy as np
from PTMCMCSampler import proposals as prop


class Base(object):
    """Base class for the testing
    """
    def setup(self, name, kwargs):
        """Setup the test class

        Parameters
        ----------
        name: str
            name of the class you want to test
        """
        jump = getattr(prop, name)
        return jump(kwargs)

    def test_2d_case(self):
        """Test the __call__ method for ndim = 2
        """
        self.samples = [10., 12.]
        new_samples, prob = self.class_object(self.samples, kwargs=None)
        assert len(new_samples) == len(self.samples)
        for i in new_samples:
            assert isinstance(i, float)

    def test_1d_case(self):
        """Test the __call__ method for ndim = 1
        """
        self.samples = [10.]
        new_samples, prob = self.class_object(self.samples, kwargs=None)
        assert len(new_samples) == len(self.samples)
        for i in new_samples:
            assert isinstance(i, float)


class TestNormal(Base):
    """Test the Normal jump proposal
    """
    def setup(self):
        """Setup the Uniform class
        """
        kwargs = {"step_size": 0.1}
        self.class_object = super(TestNormal, self).setup("Normal", kwargs)

    def test_no_kwargs(self):
        """Test to make sure that an Exception is raised when you fail to pass
        the kwargs to initalize the class
        """
        with pytest.raises(Exception) as info:
            class_object = super(TestNormal, self).setup("Normal", {})

    def test_call(self):
        """Test the __call__ method for the Uniform class
        """
        super(TestNormal, self).test_1d_case()
        super(TestNormal, self).test_2d_case()


class TestUniform(Base):
    """Test the Uniform jump proposal
    """
    def setup(self):
        """Setup the Uniform class
        """
        kwargs = {"pmin": 0.0, "pmax": 10.0}
        self.class_object = super(TestUniform, self).setup("Uniform", kwargs)

    def test_no_kwargs(self):
        """Test to make sure that an Exception is raised when you fail to pass
        the kwargs to initalize the class
        """
        with pytest.raises(Exception) as info:
            class_object = super(TestNormal, self).setup("Normal", {})

    def test_call(self):
        """Test the __call__ method for the Uniform class
        """
        super(TestUniform, self).test_1d_case()
        super(TestUniform, self).test_2d_case()
