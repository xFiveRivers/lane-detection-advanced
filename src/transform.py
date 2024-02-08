import cv2
import numpy as np

class transform():

    def __init__(self):
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
