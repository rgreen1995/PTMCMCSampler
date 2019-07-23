import pytest
from PTMCMCSampler.nompi4py import MPIDummy


class TestMPIDummp(object):
    """Test the MPIDummpy class
    """
    def setup(self):
        """Setup the MPIDummy object
        """
        self.mpidummy = MPIDummy()

    def test_Get_rank(self):
        """Test the `Get_rank` method
        """
        assert self.mpidummy.Get_rank() == 0

    def test_Get_size(self):
        """Test the `Get_size` function
        """
        assert self.mpidummy.Get_size() == 1
