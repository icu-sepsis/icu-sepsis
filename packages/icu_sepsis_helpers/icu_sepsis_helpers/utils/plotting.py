from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    window_size = np.abs(int(window_size))
    order = np.abs(int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
    # pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def moving_average(x, w):
    if w==1:
        return x
    x_padded = np.pad(x, (w//2, w-1-w//2), mode='edge')
    x_mv = np.convolve(x_padded, np.ones((w,))/w, mode='valid') 
    return x_mv

def plot_performance( *,
        window_small: int                           = 5, 
        window_large: int                           = 15,
        plot_small  : bool                          = True, 
        save        : bool                          = False, 
        save_name   : str | Path                    = 'plot.png',
        axhlines    : list[tuple[float, str, str]]  = None,
        x_lim       : tuple[float, float]           = None,
        y_lim       : tuple[float, float]           = None,
        x_label     : str                           = 'Iteration', 
        y_label     : str                           = 'Value', 
        title       : str                           = 'Performance',
        **results   : tuple[list[int], list[float]]
    ):
    # assert window_large > window_small, '`window_large` must be larger than `window_small`'
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for c, (algo_name, (ep_num_list, result_list)) in zip(colors, results.items()):
        result_filtered_small = moving_average(result_list, window_small)
        result_filtered_large = moving_average(result_list, window_large)

        if plot_small:
            ax.plot(ep_num_list, result_filtered_small, linestyle='-', color=c, alpha=0.2, linewidth=2)
            ax.plot(ep_num_list, result_filtered_large, color=c, label=algo_name)
        else:
            ax.plot(result_filtered_large, color=c, label=algo_name)
    if axhlines is not None:
        for y, style, label in axhlines:
            ax.axhline(y, linestyle=style, color='k', linewidth=0.5, label=label)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    if save:
        fig.savefig(save_name, dpi=300, bbox_inches='tight')
    else:
        fig.show()