try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    print("Failed to import matplotlib. Unable to generate the plots")
    MATPLOTLIB = False


def corner_plot(samples, **kwargs):
    """Generate a corner plot of the samples

    Parameters
    ----------
    samples: numpy.array
        array of samples
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
        return corner(samples, bins=50, **kwargs)
    return


def chains_plot(samples, **kwargs):
    """Generate a plot of the chains

    Parameters
    ----------
    samples: numpy.array
        array of samples
    **kwargs: dict
        all kwargs are passed to the matplotlib.pyplot.plot function
    """
    if MATPLOTLIB:
        fig = plt.figure()
        plt.plot(samples, **kwargs)
        plt.xlabel("Samples", fontsize=16)
        return fig
    return
