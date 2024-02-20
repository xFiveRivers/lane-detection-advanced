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
        
        # self.gray_thresh = 150  

    def apply_threshold(self, img, v_thresh, h_thresh):

        # Get HLS Colour Space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # Define upper and lower hue, lightness, and saturation values for white lines
        white_lower = np.array([int(0 / 2), int(0.78 * 255), int(0.00 * 255)])
        white_upper = np.array([int(360 / 2), int(1.00* 255), int(1.00 * 255)])

        # Find white pixels for left and right lanes
        white = cv2.inRange(hls, white_lower, white_upper)

        # Define upper and lower hue, lightness, and saturation values for yellow lines
        yellow_lower = np.array([int(40 / 2), int(0.20 * 255), int(0.20 * 255)])
        yellow_upper = np.array([int(60 / 2), int(1.00* 255), int(1.00 * 255)])

        yellow = cv2.inRange(hls, yellow_lower, yellow_upper)

        output = white | yellow
        
        # # Get colour spaces
        # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # Apply colour-specific thresholding
        # yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        # white = cv2.inRange(gray, self.white_lower, self.white_upper)

        # if np.mean(gray) <= self.gray_thresh:
        #     relative = cv2.inRange(hsv[:, :, 2], self.v_channel_lower, self.v_channel_upper)
        # else:
        #     relative = cv2.inRange(hls[:, :, 2], self.s_channel_lower, self.s_channel_upper)
        
        # output = yellow | white | relative

        # h_channel = hsv[:, :, 0]
        # v_channel = hsv[:, :, 2]

        # h_img = np.zeros_like(h_channel)
        # h_img[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

        # v_img = np.zeros_like(v_channel)
        # v_img[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

        # output = v_img | h_img

        return output