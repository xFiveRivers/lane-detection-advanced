import cv2
import numpy as np

class Threshold():
    def __init__(self):
        pass

    def threshold(img):
        thresh = (200,255)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        output = np.zeros_like(v_channel)
        output[(v_channel >= thresh[0]) & (v_channel <= thresh[1])] = 1

        return output

