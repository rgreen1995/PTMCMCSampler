
from . import plot

class Result(object):
    """Class to handle the samples
    """
    def __init__(self, samples):
        self.samples = samples

    def save(self, outfile=None, outdir=None):
        """Save the samples to file
        """

    def _plot(self, func):
        """Generate plots according to a specific function

        Parameters
        ----------
        func: function
            plotting function to execute
        """
        return func(self.samples)

    @property
    def plot_acceptance_rate(self):
        """Generate a plot for the acceptance rate
        """
        fig = self._plot(plot.acceptance_rate)
        return fig

    @property
    def plot_chains(self):
        """Generate a plot of the chains
        """
        fig = self._plot(plot.chains)
        return fig

    @property
    def plot_corner(self):
