import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    MATPLOTLIB = True
except:
    print("Failed to import matplotlib. Unable to generate the plots")
    MATPLOTLIB = False


def get_colors(n):
    """Return a color wheel of n dimensions

    Parameters
    ----------
    n: float
        the number of dimensions of your color wheel
    """
    color=cm.rainbow(np.linspace(0, 1, n))
    return color


def corner_plot(samples, num_chains=1, **kwargs):
    """Generate a corner plot of the samples

    Parameters
    ----------
    samples: numpy.array
        array of samples
    num_chains: int
        the number of cold chains stored in samples
    **kwargs: dict
        all kwargs are passed to the corner python package
    """
    if MATPLOTLIB:
        try:
            from corner import corner
        except:
            print("Failed to import 'corner'. A corner plot cannot be "
                  "generated")
            return
        colors = get_colors(num_chains)
        fig = corner(samples[0], bins=50, color=colors[0], **kwargs)
        for i in range(1, num_chains):
            _ = corner(samples[i], bins=50, fig=fig, color=colors[i], **kwargs)
        return fig
    return


def chains_plot(samples, num_chains=1, **kwargs):
    """Generate a plot of the chains

    Parameters
    ----------
    samples: numpy.array
        array of samples
    num_chains: int
        the number of cold chains stored in samples
    **kwargs: dict
        all kwargs are passed to the matplotlib.pyplot.plot function
    """
    if MATPLOTLIB:
        fig = plt.figure()
        colors = get_colors(num_chains)
        for i in range(num_chains):
            plt.plot(
                samples[i], label="chain_%s" % (i), color=colors[i], **kwargs)
        plt.xlabel("Samples", fontsize=16)
        plt.legend(loc="best")
        return fig
    return
