import cv2
import numpy as np

class Threshold():
    def __init__(self):
        # Specify yellow threshold range
        self.yellow_lower = np.array([0, 100, 100])
        self.yellow_upper = np.array([50, 255, 255])

        # Specify white threshold range
        self.white_lower = np.array([215])
        self.white_upper = np.array([255])

        # Specify value channel threshold range
        self.v_channel_lower = np.array([170])
        self.v_channel_upper = np.array([255]) 

        self.s_channel_lower = np.array([200])
        self.s_channel_upper = np.array([255]) 
        
        self.gray_thresh = 125     

    def apply_threshold(self, img, v_thresh, h_thresh):
        
        # Get colour spaces
        hls = cv2.cvtColor(img, cv2.BGR2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply colour-specific thresholding
        yellow = cv2.inRange(hsv, self.yellow_lower_range, self.yellow_upper_range)
        white = cv2.inRange(gray, self.white_lower_range, self.white_upper_range)

        if np.mean(gray) <= self.gray_thresh:
            relative = cv2.inRange(hsv[:, :, 2], self.v_channel_lower_range, self.v_channel_upper_range)
        else:
            relative = cv2.inRange(hls[:, :, 2], self.s_channel_lower, self.s_channel_upper)
        
        output = yellow | white | relative

        # h_channel = hsv[:, :, 0]
        # v_channel = hsv[:, :, 2]

        # h_img = np.zeros_like(h_channel)
        # h_img[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

        # v_img = np.zeros_like(v_channel)
        # v_img[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

        # output = v_img | h_img

        return output