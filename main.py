import cv2
import numpy as np
from moviepy.editor import VideoFileClip

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

        return out_img
    
    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4)) 

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), 24, (frame_width, frame_height))

        if cap.isOpened() == False:
            return 'Error openeing video.'
        
        i = 0

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                processed_frame = self.detect(frame)
                # print(i)
                out.write(processed_frame)
                i += 1           
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif i == 120:
                    break
            else:
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()


    def process_image(self, input_path, output_path):
        input_img = cv2.imread(input_path)
        output_img = self.detect(input_img)
        cv2.imwrite(output_path, output_img)

