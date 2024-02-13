import cv2
import numpy as np

class Threshold():
    def __init__(self):
        pass

    def apply_threshold(self, img, v_thresh, h_thresh):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        v_channel = hsv[:, :, 2]

        h_img = np.zeros_like(v_channel)
        h_img[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

        v_img = np.zeros_like(v_channel)
        v_img[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

        return h_img | v_img