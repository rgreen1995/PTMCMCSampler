import pytest
import matplotlib
import numpy as np
from PTMCMCSampler import plots


def test_get_colors():
    """Test the `get_colors` function
    """
    n = 10
    colors = plots.get_colors(n)
    assert len(colors) == n

def test_chain_plots():
    """Test that we can produce a chain plot
    """
    samples = np.array([np.random.random(2) for i in range(500)])
    assert isinstance(plots.chains_plot(samples), matplotlib.figure.Figure)


def test_corner_plots():
    """Test that we can produce a corner plot
    """
    samples = np.array([np.random.random(2) for i in range(500)])
    assert isinstance(plots.corner_plot(samples), matplotlib.figure.Figure)
    assert isinstance(
        plots.corner_plot(samples, truths=[0.5, 0.1]), matplotlib.figure.Figure)
