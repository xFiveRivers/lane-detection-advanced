import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.calibration import *
from src.transform import *


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


def plot_channels(dict: dict, title: str, cmap: str = 'gray', nrows: int = 1, ncols: int = 3, figsize: tuple = (12, 3)):
    """Plots the colour channels of a given colour space.

    Parameters
    ----------
    dict : dict
        A dictionary containing channel titles (keys) and arrays (values).
    title : str
        The title to display on the plot.
    cmap : str
        The colourmap as per matplotlib's standard.
    nrows : int, optional
        Number of rows in the plot, by default 1.
    ncols : int, optional
        Number of columns (number of channels) in the plot, by default 3.
    figsize : tuple, optional
        Figure size in inches, by default (12, 3)
    """

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if ncols > 1:
        axs = axs.flatten()
        for channel, ax in zip(dict, axs):
            ax.imshow(dict[channel], cmap=cmap)
            ax.title.set_text(channel)
            ax.axis(False)
    else:
        axs.imshow(list(dict.values())[0], cmap=cmap)
    fig.suptitle(title)
    plt.show()


def get_bev(src_path: str):
    """Returns a list of bird's-eye-view of front-view frames.

    Parameters
    ----------
    src_path : str
        Path to the folder containing the frames. 

    Returns
    -------
    list
        A list containing the trasnformed images.
    """

    # Initialize classes
    calibration = Calibration('camera_cal', (9, 6))
    transform = Transform()

    # Initialize variables
    bev_list = []

    # Get file names
    fnames = glob("{}/*".format(src_path))

    # Loop through files, apply calibration, and transform
    for file in fnames:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = calibration.undistort(img)
        img = transform.orig_to_bev(img)
        bev_list.append(img)

    return bev_list


def plot_row_images(img_list: list, title_list: list, cmap: str = None):
    """Plots a single row of images.

    Parameters
    ----------
    img_list : list
        List of images to plot.
    title_list : list
        List of subtitles for each image.
    cmap : str, optional
        Keyword argument for matplotlib image rendering, by default None.
    """

    # Create figure with subplots
    fig, axs = plt.subplots(nrows=1, ncols=len(img_list), figsize=(12,6))
    
    # Show image in each subplot
    for i, img, in enumerate(img_list):
        axs[i].imshow(img_list[i], cmap=cmap)
        axs[i].axis('off')
        axs[i].title.set_text(title_list[i])
    
    # Show figure
    plt.show()


def binary_threshold(img, thresh: int):
    """Performs binary thresholding on a grayscale image.

    Parameters
    ----------
    img : array-like
        Grayscale image to threshold.
    thresh : int
        Grayscale threshold value.

    Returns
    -------
    binary : array-like
        Binary image from grayscale thresholding.
    """

    # Create black image 
    binary = np.zeros_like(img)

    # Threshold pixels
    binary[img < thresh] = 0
    binary[img >= thresh] = 1

    return binary