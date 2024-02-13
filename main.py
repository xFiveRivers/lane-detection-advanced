import cv2
import numpy as np

from src.calibration import *
from src.transform import *
from src.threshold import *
from src.lines import *

class LaneDetection:

    def __init__(self):
        self.calibration = Calibration('camera_cal', (9, 6))
        self.tranform = Transform()
        self.threshold = Threshold()
        self.lines = Lines()

    def detect(self, img):
        orig_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.tranform.orig_to_bev(img)
        img = self.threshold.apply_threshold(img, (210, 255), (20, 30))
        img = self.lines.sliding_window(img)
        img = self.tranform.bev_to_orig(img)

        out_img = cv2.addWeighted(orig_img, 0.8, img, 1.0, 0.0)

    
        