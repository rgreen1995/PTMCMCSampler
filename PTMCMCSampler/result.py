import numpy as np
from . import plots

class Result(object):
    """Class to handle the samples from  the MCMC chains

    Parameters
    ----------
    initial_samples: numpy.array
        array of samples that have not been burnt in
    """
    def __init__(self, initial_samples, jump_proposal_name=None):
        self.initial_samples = initial_samples
        self.burnin = 0
        self.jump_proposal_name = jump_proposal_name

    def save(self, outfile=None, outdir="./"):
        """Save the samples to file

        Parameters
        ----------
        outfile: str
            The name of the file that you wish to save the samples as
        outdir: str
            The path to the directory to store the samples. Default = './'
        """
        if outfile is None and self.jump_proposal_name is None:
            outfile = "%s/chains.txt" % (outdir)
        elif outfile is None:
            outfile = "%s/%s_chains.txt" % (outdir, self.jump_proposal_name)
        np.savetxt(outfile, self.initial_samples)

    def _plot(self, func, **kwargs):
        """Generate plots according to a specific function

        Parameters
        ----------
        func: function
            plotting function to execute
        """
        return func(self.samples, **kwargs)

    def set_burnin(self, burnin=None):
        """Set the burnin of the chain

        burnin: int
            Number of samples to discard from burnin. If not provided, the
            the default is 25% of the chain length
        """
        if burnin is None:
            burnin = int(0.25*len(samples))
        elif isinstance(burnin, float):
            burnin = int(burnin)
        setattr(self, "burnin", burnin)

    @property
    def samples(self):
        """Return the samples
        """
        return self.initial_samples[self.burnin:]

    def plot_chains(self):
        """Generate a plot of the chains
        """
        fig = self._plot(plots.chains_plot)
        return fig

    def plot_corner(self, truths=None):
        """Generate a corner plot of the samples
        """
        fig = self._plot(plots.corner_plot, truths=truths)
        return fig
