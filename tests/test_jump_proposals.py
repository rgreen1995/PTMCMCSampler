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

    def test_1d_case(self):
        """Test the __call__ method for ndim = 1
        """
        if self.kwargs and "groups" in list(self.kwargs.keys()):
            self.kwargs["groups"] = [np.array([0])]
        if self.kwargs and "DEBuffer" in list(self.kwargs.keys()):
            self.kwargs["DEBuffer"] = np.array([
                np.random.random(1) for i in range(500)])
        if self.kwargs and "U" in list(self.kwargs.keys()):
            self.kwargs["U"] = [np.array([[1.]])]
            self.kwargs["S"] = [np.array([0.01])]
        if self.kwargs and "mm_inv" in list(self.kwargs.keys()):
            self.kwargs["mm_inv"] = np.eye(1) * 0.1**2
        self.samples = np.array([10.])
        if hasattr(self, "class_object1d"):
            new_samples, prob = self.class_object1d(self.samples, kwargs=self.kwargs)
        else:
            new_samples, prob = self.class_object(self.samples, kwargs=self.kwargs)
        assert len(new_samples) == len(self.samples)
        for i in new_samples:
            assert isinstance(i, float)

    def test_2d_case(self):
        """Test the __call__ method for ndim = 2
        """
        self.samples = np.array([10., 12.])
        if hasattr(self, "class_object2d"):
            new_samples, prob = self.class_object2d(self.samples, kwargs=self.kwargs)
        else:
            new_samples, prob = self.class_object(self.samples, kwargs=self.kwargs)
        assert len(new_samples) == len(self.samples)
        for i in new_samples:
            assert isinstance(i, float)


class BaseAdaptiveGaussian(Base):
    """Class to setup variables for all Adaptive Gaussian jump proposals
    """
    def AdaptiveGaussian_variables(self):
        """Setup the variables for all adaptive Gaussian classes
        """
        self.naccepted = 100
        self.iter = 1
        self.chain = np.array([[1., 2.], [3., 4.]])
        self.kwargs = {
            "naccepted": self.naccepted,
            "iter": self.iter,
            "chain": self.chain}


class BaseAdaptiveCovariance(Base):
    """Class to setup variables for all Adaptive Covariance jump proposals
    """
    def AdaptiveCovariance_variables(self):
        """Setup the variables for all adaptive covariance classes
        """
        self.groups = [np.array([0, 1])]
        self.beta = 1.0
        self.U = [np.array([[1., 0.], [0., 1.]])]
        self.S = [np.array([0.01, 0.01])]
        self.kwargs = {
            "groups": self.groups,
            "beta": self.beta,
            "U": self.U,
            "S": self.S}


class BaseGradientClass(Base):
    """Class to setup variables for all Gradient jump proposals
    """
    def Gradient_variables(self):
        """Setup the variables for all Gradient classes
        """
        self.loglik_grad = self.loglik_grad_func
        self.logprior_grad = self.logprior_grad_func
        self.mm_inv = np.eye(2) * 0.1**2
        self.nburn = 500
        self.kwargs = {
            "loglik_grad": self.loglik_grad,
            "logprior_grad": self.logprior_grad,
            "mm_inv": self.mm_inv,
            "nburn": self.nburn}

    def loglik_grad_func(self, x):
        """Basic log likelihood function
        """
        ll = -0.5*np.sum(x**2)-len(x)*0.5*np.log(2*np.pi)
        ll_grad = -x
        return ll, ll_grad

    def logprior_grad_func(self, x):
        """Basic log prior function
        """
        if np.all(0. <= x) and np.all(10. >= x):
            z = 0.0
        else:
            z = -np.inf
        return z, np.zeros_like(x)


class TestMALA(BaseGradientClass):
    """Test the MALA jump proposal
    """
    def setup(self):
        """Setup the MALA class
        """
        self.Gradient_variables()
        self.class_object2d = super(
            TestMALA, self).setup("MALA", self.kwargs)
        self.kwargs["mm_inv"] = np.eye(1) * 0.1**2
        self.class_object1d = super(
            TestMALA, self).setup("MALA", self.kwargs)

    def test_call(self):
        """Test the __call__ method for the MALA class
        """
        super(TestMALA, self).test_1d_case()
        super(TestMALA, self).test_2d_case()


class TestHMC(BaseGradientClass):
    """Test the HMC jump proposal
    """
    def setup(self):
        """Setup the HMC class
        """
        self.Gradient_variables()
        self.kwargs["step_size"] = 0.01
        self.kwargs["nminsteps"] = 0
        self.kwargs["nmaxsteps"] = 300
        self.class_object2d = super(
            TestHMC, self).setup("HMC", self.kwargs)
        self.kwargs["mm_inv"] = np.eye(1) * 0.1**2
        self.class_object1d = super(
            TestHMC, self).setup("HMC", self.kwargs)

    def test_no_kwargs(self):
        """Test to make sure that an Exception is raised when you fail to pass
        the kwargs to initalize the class
        """
        with pytest.raises(Exception) as info:
            class_object = super(TestHMC, self).setup("HMC", {})

    def test_call(self):
        """Test the __call__ method for the HMC class
        """
        super(TestHMC, self).test_1d_case()
        super(TestHMC, self).test_2d_case()


class TestDifferentialEvolution(Base):
    """Test the Differential Evolution jump proposal
    """
    def setup(self):
        """Setup the DifferentialEvolution class
        """
        self.class_object = super(
            TestDifferentialEvolution, self).setup(
            "DifferentialEvolution", {})
        self.groups = [np.array([0, 1])]
        self.beta = 1.0
        self.DEBuffer = np.array([np.random.random(2) for i in range(500)])
        self.kwargs = {
            "groups": self.groups,
            "beta": self.beta,
            "DEBuffer": self.DEBuffer}

    def test_call(self):
        """Test the __call_ method for the DifferentialEvolution class
        """
        super(TestDifferentialEvolution, self).test_1d_case()
        super(TestDifferentialEvolution, self).test_2d_case()


class TestSingleComponentAdaptiveCovariance(BaseAdaptiveCovariance):
    """Test the Single Component Adaptive Covariance jump proposal
    """
    def setup(self):
        """Setup the SingleComponentAdaptiveCovariance class
        """
        self.class_object = super(
            TestSingleComponentAdaptiveCovariance, self).setup(
            "SingleComponentAdaptiveCovariance", {})
        self.AdaptiveCovariance_variables()

    def test_call(self):
        """Test the __call_ method for the SingleComponentAdaptiveCovariance
        class
        """
        super(TestSingleComponentAdaptiveCovariance, self).test_1d_case()
        super(TestSingleComponentAdaptiveCovariance, self).test_2d_case()


class TestSingleComponentAdaptiveGaussian(BaseAdaptiveGaussian):
    """Test the Single Component Adaptive Gaussian jump proposal
    """
    def setup(self):
        """Setup the SingleComponentAdaptiveGaussian class
        """
        self.class_object = super(
            TestSingleComponentAdaptiveGaussian, self).setup(
            "SingleComponentAdaptiveGaussian", {})
        self.AdaptiveGaussian_variables()

    def test_call(self):
        """Test the __call__ method for the SingleComponentAdaptiveGaussian
        class
        """
        super(TestSingleComponentAdaptiveGaussian, self).test_1d_case()
        super(TestSingleComponentAdaptiveGaussian, self).test_2d_case()


class TestMultiComponentAdaptiveGaussian(BaseAdaptiveGaussian):
    """Test the Multi Component Adaptive Gaussian jump proposal
    """
    def setup(self):
        """Setup the MultiComponentAdaptiveGaussian class
        """
        self.class_object = super(
            TestMultiComponentAdaptiveGaussian, self).setup(
            "MultiComponentAdaptiveGaussian", {})
        self.AdaptiveGaussian_variables()

    def test_call(self):
        """Test the __call__ method for the MultiComponentAdaptiveGaussian
        class
        """
        super(TestMultiComponentAdaptiveGaussian, self).test_1d_case()
        super(TestMultiComponentAdaptiveGaussian, self).test_2d_case()


class TestAdaptiveGaussian(BaseAdaptiveGaussian):
    """Test the Adaptive Gaussian jump proposal
    """
    def setup(self):
        """Setup the AdaptiveGaussian class
        """
        self.class_object = super(
            TestAdaptiveGaussian, self).setup(
            "AdaptiveGaussian", {})
        self.AdaptiveGaussian_variables()

    def test_call(self):
        """Test the __call__ method for the AdaptiveGaussian class
        """
        super(TestAdaptiveGaussian, self).test_1d_case()
        super(TestAdaptiveGaussian, self).test_2d_case()


class TestNormal(Base):
    """Test the Normal jump proposal
    """
    def setup(self):
        """Setup the Uniform class
        """
        kwargs = {"step_size": 0.1}
        self.class_object = super(TestNormal, self).setup("Normal", kwargs)
        self.kwargs = None

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
        self.kwargs = None

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
