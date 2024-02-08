import cv2
import numpy as np

class transform():

    def __init__(self):
        """Class used to transform images from front to bird's-eye views.
        
        Attributes
        ----------
        src : array-like
            The ROI points on the front-view image.
        dst : array-like
            The destination points for the bird's-eye view.
        matrix : array-like
            Matrix to transform from front to bird's-eye.
        matrix_inv : array-like
            Matrix to transform from bird's-eye to front. 
        """
        self.src = np.int32([
            (550, 450),
            (250, 650),
            (1175, 650),
            (750, 450)
        ])

        self.dst = np.float32([
            (100, 0),
            (100, 750),
            (1100, 750),
            (1100, 0)
        ])

        self.matrix = cv2.getPerspectiveTransform(self.src, self.dst)
        self.matrix_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def orig_to_bev(self, img):
        """Transforms an image to bird's-eye view.

        Only a pre-specified region of the image will be transformed and returned.

        Parameters
        ----------
        img : array-like
            The image to be transformed.

        Returns
        -------
        array-like
            The bird's-eye view of the region.
        """
        result = cv2.warpPerspective(img, self.matrix, (img.shape[1], img.shape[0]))
        return result
    
    def bev_to_orig(self, img):
        """Transforms a bird's-eye view to a front-view.

        Parameters
        ----------
        img : array-like
            The transformed bird's-eye view.

        Returns
        -------
        array-like
            The transformed front-view.
        """
        result = cv2.warpPerspective(img, self.matrix_inv, (img.shape[1], img.shape[0]))
        return result
