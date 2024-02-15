import cv2
import numpy as np

class Threshold():
    def __init__(self):
        pass

    def apply_threshold(self, img, v_thresh, h_thresh):
        # Specify yellow threshold range
        yellow_lower_range = np.array([0, 100, 100])
        yellow_upper_range = np.array([50, 255, 255])

        # Specify white threshold range
        white_lower_range = np.array([215])
        white_upper_range = np.array([255])

        # Get colour spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply colour-specific thresholding
        yellow = cv2.inRange(hsv, yellow_lower_range, yellow_upper_range)
        white = cv2.inRange(gray, white_lower_range, white_upper_range)
        
        output = yellow | white

        # h_channel = hsv[:, :, 0]
        # v_channel = hsv[:, :, 2]

        # h_img = np.zeros_like(h_channel)
        # h_img[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

        # v_img = np.zeros_like(v_channel)
        # v_img[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

        # output = v_img | h_img

        return output