import cv2
import numpy as np

class Threshold():

    def __init__(self):
        """Class used to run pipeline.

        Attributes
        ----------
        lightness_lower: array-like
            The lower bound of the lightness threshold range
        lightness_upper: array-like
            The upper bound of the lightness threshold range
        yellow_lower: array-like
            The lower bound of the HSV colour space
        yellow_upper: array-like
            The upper bound of the HSV colour space
        """

        # Define upper and lower bounds of lightness range
        self.lightness_lower = np.array([195])
        self.lightness_upper = np.array([255])

        # Define upper and lower bounds of HSV space
        self.yellow_lower = np.array([int(40 / 2), int(0.00 * 255), int(0.00 * 255)])
        self.yellow_upper = np.array([int(50 / 2), int(1.00 * 255), int(1.00* 255)])
        

    def apply_threshold(self, img):
        """Applies thresholding to an image for line detection.

        Thresholds the lightness channel of the HLS colour space for white lines
        and thresholds the HSV colour space for yellow lines.

        Parameters
        ----------
        img : array-like
            Source bird's-eye view image.

        Returns
        -------
        output : array-like
            Output binary image containing thresholded pixels.
        """

        # Get colourspaces
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Get filtered images
        lightness = cv2.inRange(hls[:, :, 1], self.lightness_lower, self.lightness_upper)
        yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        # Combine filtered images
        output = lightness | yellow

        return output