import cv2
import numpy as np
import os
import shutil

from src.calibration import *
from src.transform import *
from src.threshold import *
from src.lines import *


class LaneDetection:

    def __init__(self):
        """Class for lane detection pipeline.

        Attributes
        ----------
        calibration : class
            Initialized class for camera calibration.
        transform : class
            Initialized class for perspective transformation.
        threshold : class
            Initialized class for image thresholding.
        lines : class
            Initialized class for fitting lines.
        """

        self.calibration = Calibration('camera_cal', (9, 6))
        self.tranform = Transform()
        self.threshold = Threshold()
        self.lines = Lines()


    def detect(self, img, draw_boxes):
        """Pipeline function for finding and drawing lane lines.

        Parameters
        ----------
        img : array-like
            Source image.
        draw_boxes : boolean
            Flag for drawing windows.

        Returns
        -------
        out_img
            Output image with lines drawn on original image.
        """

        orig_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.tranform.orig_to_bev(img)
        img = self.threshold.apply_threshold(img, (210, 255), (20, 30))
        img = self.lines.sliding_window(img, draw_boxes)
        img = self.tranform.bev_to_orig(img)
        out_img = cv2.addWeighted(orig_img, 0.8, img, 1.0, 0.0)

        return out_img
    

    def process_video(self, input_path: str, output_path: str, debug=False, draw_boxes=False):
        """Detect lane lines in a video.

        Parameters
        ----------
        input_path : str
            Path from root to video including file name.
        output_path : str
            Path from root to save output video including file name.
        debug : bool, optional
            Flag for saving individual frames for debugging, by default False.
        draw_boxes : bool, optional
            Flag for drawing windows, by default False.
        """
        # Create video caputre object
        cap = cv2.VideoCapture(input_path)

        # Get frame dimensions
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4)) 

        # Create video writer object
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), 24, (frame_width, frame_height))

        # Check if video can be opened
        if cap.isOpened() == False:
            return 'Error openeing video.'
        
        # Initialize debug parameters
        if debug == True:
            # Path for individual frames
            solo_path = 'output_media/debug/output_frames/solo'
            # Path for side-by-side frames
            sbs_path = 'output_media/debug/output_frames/side-by-side'

            # Delete folders
            shutil.rmtree(solo_path)
            shutil.rmtree(sbs_path)

            # Create folders
            os.mkdir(solo_path)
            os.mkdir(sbs_path)
            
            # Frame counter
            i = 0

        # Loop through video frame by frame
        while(cap.isOpened()):
            # Read frame
            ret, frame = cap.read()

            if ret == True:
                # Run pipeline on frame
                processed_frame = self.detect(frame, draw_boxes)

                # Ouput processed frame to video write object
                out.write(processed_frame)

                # Save debug frames
                if debug == True:
                    cv2.imwrite(f'{solo_path}/{i}_frame.png', frame)
                    cv2.imwrite(f'{sbs_path}/{i}_side-by-side.png', np.concatenate((frame, processed_frame), axis=1))
                    i += 1           

                # Keypress break operation
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release and destroy all OpenCV objects
        cap.release()
        out.release()
        cv2.destroyAllWindows()


    def process_image(self, input_path: str, output_path: str, draw_boxes=False):
        """Detect lane lines for an image.

        Parameters
        ----------
        input_path : str
            Path from root to source image with file name.
        output_path : str
            Path from to target image with file name.
        draw_boxes : bool, optional
            Flag for drawing windows, by default False.
        """

        # Read image
        input_img = cv2.imread(input_path)

        # Run pipeline on image
        output_img = self.detect(input_img, draw_boxes)

        # Save image
        cv2.imwrite(output_path, output_img)