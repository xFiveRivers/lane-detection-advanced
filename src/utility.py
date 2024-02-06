import cv2
import numpy as np

def draw_roi(img, points, color):
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