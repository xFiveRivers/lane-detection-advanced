import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_channels(dict, title, nrows=1, ncols=3, figsize=(12, 3)):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()
    for channel, ax in zip(dict, axs):
        ax.imshow(dict[channel], cmap='gray')
        ax.title.set_text(channel)
        ax.axis(False)
    fig.suptitle(title)
    plt.show()