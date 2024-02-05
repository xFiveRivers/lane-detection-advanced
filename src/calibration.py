import cv2
import numpy as np
from glob import glob

class Calibration():

    def __init__(self, cal_folder, board_size):
        self.mtx = None
        self.dist = None

        imgpoints = []
        objpoints = []

        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[:board_size[0], :board_size[1]].T.reshape(-1,2)

        for file in fnames:
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)

            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
        
        img_shape = gray.shape[::-1]

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

        if not ret:
            print('Camera calibration not possible.')

    def undistort(img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)