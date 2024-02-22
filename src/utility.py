import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_roi(img: np.ndarray, points: np.ndarray, color: tuple = (255, 0, 0)):
    """Draws the boundry of a region of interest on the image

    Parameters
    ----------
    img : array_like
        The image with the boundried to be drawn on
    points : array_like
        An array containing the the x and y coordinates of the
        vertices for the boundry
        Example: np.array([[(x1, y1), (x2, y2), (x3, y3)]])
    color : tuple
        A tuple containing the RGB values (respectively) for the boundry
        color

    Returns
    -------
    result : array_like
        The original image with the drawn ROI boundries
    """
    for point in points:
        for x, y in point:
            cv2.circle(img, (x, y), radius = 10, color = color, thickness = -1)

    result = cv2.polylines(img, points, True, color, thickness=5)

    return result

def plot_channels(dict: dict, title: str, nrows: int = 1, ncols: int = 3, figsize: tuple = (12, 3)):
    """Plots the colour channels of a given colour space.

    Parameters
    ----------
    dict : dict
        A dictionary containing channel titles (keys) and arrays (values).
    title : str
        The title to display on the plot.
    nrows : int, optional
        Number of rows in the plot, by default 1.
    ncols : int, optional
        Number of columns (number of channels) in the plot, by default 3.
    figsize : tuple, optional
        Figure size in inches, by default (12, 3)
    """
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()
    for channel, ax in zip(dict, axs):
        ax.imshow(dict[channel], cmap='gray')
        ax.title.set_text(channel)
        ax.axis(False)
    fig.suptitle(title)
    plt.show()