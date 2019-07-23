import numpy as np
from . import plots

try :
    import arviz as az
except ImportError:
    print("Do not have arviz package")
    pass

class Result(object):
    """Class to handle the samples from  the MCMC chains

    Parameters
    ----------
    initial_samples: numpy.array
        array of samples that have not been burnt in
    """

    def __init__(
        self,
        initial_samples,
        initial_likelihood_vals=None,
        initial_prior_vals=None,
        burnin=0,
        num_chains=2,
        jump_proposal_name=None,
    ):
        self.initial_samples = initial_samples
        self.inital_likelihood_vals = initial_likelihood_vals
        self.initial_prior_vals = initial_prior_vals
        self.burnin = burnin
        self.jump_proposal_name = jump_proposal_name
        self.num_chains = num_chains

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
            outfile = "%s/chains" % (outdir)
        elif outfile is None:
            outfile = "%s/%s_chains" % (outdir, self.jump_proposal_name)
        if ".txt" in outfile:
            outfile = outfile.split(".txt")[0]
        for i in range(self.num_chains):
            np.savetxt("%s_chain%s.txt" % (outfile, i),  self.initial_samples[i])

    def _plot(self, func, **kwargs):
        """Generate plots according to a specific function

        Parameters
        ----------
        func: function
            plotting function to execute
        """
        return func(self.samples, **kwargs)

    def set_burnin(self, burnin):
        """Set the burnin of the chain

        burnin: int
            Number of samples to discard from burnin. If not provided, the
            the default is 25% of the chain length
        """
        if burnin is None:
            burnin = int(0.25 * len(self.initial_samples))
        elif isinstance(burnin, float):
            burnin = int(burnin)
        setattr(self, "burnin", burnin)

    @property
    def samples(self):
        """Return the samples
        """
        if self.num_chains == 1:
            np.expand_dims(self.initial_samples, axis=0)
        return self.initial_samples[:, self.burnin :]

    @property
    def likelihood_values(self):
        """Return the likelihood values for each sample
        """
        if self.num_chains == 1:
            np.expand_dims(self.inital_likelihood_vals, axis=0)
        return self.inital_likelihood_vals[:, self.burnin :]

    @property
    def prior_values(self):
        """Return the prior values for each sample
        """
        if self.num_chains == 1:
            np.expand_dims(self.initial_prior_vals, axis=0)
        return self.initial_prior_vals[:, self.burnin :]

    @property
    def effective_sample_size(self):
        try :
            arviz_samples = az.convert_to_inference_data(self.samples)
            ess = az.ess(arviz_samples)
        except ModuleNotFoundError :
            print('Summary relies on arviz and arviz is not instaled')
            ess = None
        return ess

    def summary(self):
        """ Returns a summary of the sample statistics
        """
        try :
            summary = az.summary(self.samples, credible_interval=0.9)
        except ModuleNotFoundError :
            print('caclulating ess relies on arviz and arviz is not instaled')
            summary = None
        return summary

    def plot_chains(self):
        """Generate a plot of the chains
        """
        fig = self._plot(plots.chains_plot, num_chains=self.num_chains)
        return fig

    def plot_corner(self, truths=None):
        """Generate a corner plot of the samples
        """
        fig = self._plot(plots.corner_plot, num_chains=self.num_chains, truths=truths)
        return fig
