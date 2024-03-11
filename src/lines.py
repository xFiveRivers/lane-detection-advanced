import cv2
import numpy as np

class Lines():

    def __init__(self):
        """Class used to detect the lane lines in an image

        Attributes
        ----------
        img : array-like
            Source image.
        out_img : array-like
            Ouput image of fitted lines.
        non_zero : array-like
            Matrix of x and y coordinates of non-zero pixels.
        non_zero_x : array-like
            Array containing x-coordinates of non-zero pixels.
        non_zero_y : array-like
            Array containing y-coordinates of non-zero pixels.
        left_x_curr : int
            Pointer for left window center x-coordinate.
        right_x_curr : int
            Pointer for right window center x-coordinate.
        y_curr : int
            Point for window center y-coordinate.
        window_height : int
            Window height in pixels.
        top_left : tuple
            Tuple for window top-left corner pointer x and y coordinates.
        bottom_right : tuple
            Tuple for window bottom-right corner pointer x and y coordinates.
        left_fit : array-like
            Left line 2nd order polynomial coefficients.
        right_fit : array-like
            Right line 2nd order polynomial coefficients.
        margin : int
            One half of total window width.
        min_pixels : int
            Minimum number of pixels in window to update x_curr pointer.
        n_windows : int
            Total number of windows.
        """

        # Images
        self.img = None
        self.out_img = None

        # Non-zero matrices
        self.non_zero = None
        self.non_zero_x = None
        self.non_zero_y = None

        # Window center pointers
        self.left_x_curr = None
        self.right_x_curr = None
        self.y_curr = None

        # Window parameters
        self.window_height = None
        self.top_left = None
        self.bottom_right = None
        
        # Fitted Lines
        self.left_fit = None
        self.right_fit = None

        # Hyperparameters
        self.margin = 150
        self.min_pixels = 25
        self.n_windows = 9


    def draw_lines(self, left_x, left_y, right_x, right_y):
        """_summary_

        Parameters
        ----------
        left_x : array-like
            X-coordinates of all detected pixels for the left line.
        left_y : array-like
            Y-coordinates of all detected pixels for the left line.
        right_x : array-like
            X-coordinates of all detected pixels for the right line.
        right_y : array-like
            Y-coordinates of all detected pixels for the right line.

        Returns
        -------
        array-like
            Output image of fitted lines.
        """

        if len(left_x) >= 1000:
            self.left_fit = np.polyfit(left_y, left_x, 2)
        if len(right_x) >= 1000:
            self.right_fit = np.polyfit(right_y, right_x, 2)

        draw_x = np.linspace(0, self.img.shape[1], self.img.shape[1])

        draw_y_left = np.polyval(self.left_fit, draw_x)
        draw_y_right = np.polyval(self.right_fit, draw_x)

        draw_points_left = (np.asarray([draw_y_left, draw_x]).T).astype(np.int32)
        draw_points_right = (np.asarray([draw_y_right, draw_x]).T).astype(np.int32)

        # out_img = np.zeros((self.img.shape[0], self.img.shape[1], 3))

        cv2.polylines(self.out_img, [draw_points_left], False, (0, 255, 0), thickness=20)
        cv2.polylines(self.out_img, [draw_points_right], False, (0, 255, 0), thickness=20)
        
        return (self.out_img).astype(np.uint8)


    def get_features(self):
        """Initializes parameters for aliding window."""

        # Get height of windows
        self.window_height = int(self.img.shape[0] // self.n_windows)

        # Create historgram
        hist = np.sum(self.img, axis=0)

        # Define starting x-coordinates of left and right windows
        left_x_base = np.argmax(hist[:600])
        right_x_base = np.argmax(hist[800:1200]) + 800

        # Create pointers for window center coordinates
        self.left_x_curr = left_x_base
        self.right_x_curr = right_x_base
        self.y_curr = self.img.shape[0] - self.window_height // 2

        # Get x and y poisitions of all non-zero pixels in image
        self.non_zero = self.img.nonzero()
        self.non_zero_x = np.array(self.non_zero[1])
        self.non_zero_y = np.array(self.non_zero[0])


    def find_pixels(self, center):
        """Finds coordinates of pixels within the window.

        Parameters
        ----------
        center : tuple
            X and y coordinates for the center of the window.

        Returns
        -------
        targets_x : array-like
            X-coordinates of detected pixels in window.
        target_y : array-like
            Y-coordinates of detected pixels in window.

        """
        # Define top-left and bottom-right x and y coords for window boundry
        self.top_left = (center[0] - self.margin, center[1] - (self.window_height // 2))
        self.bottom_right = (center[0] + self.margin, center[1] + self.window_height // 2)
        
        # Create conditions for array indexing
        cond_x = (self.non_zero_x >= self.top_left[0]) & (self.non_zero_x <= self.bottom_right[0])
        cond_y = (self.non_zero_y >= self.top_left[1]) & (self.non_zero_y <= self.bottom_right[1])

        # Index non-zero array to only take pixels within the window boundries
        targets_x = self.non_zero_x[cond_x & cond_y]
        targets_y = self.non_zero_y[cond_x & cond_y]

        return targets_x, targets_y
    

    def sliding_window(self, img, draw_boxes=False):
        """Performs sliding window algorithm.

        Parameters
        ----------
        img : array-like
            Source threshold image.
        draw_boxes : bool, optional
            Flag for drawing windows, by default False.

        Returns
        -------
        out_img : array-like
            Output image of fitted lines.
        """
        self.img = img
        self.out_img = np.zeros((self.img.shape[0], self.img.shape[1], 3))

        # Get image parameters for sliding window
        self.get_features()

        # Create empty lists to store lane pixel locations
        left_x, left_y, right_x, right_y = [], [], [], []

        # Start sliding window
        for _ in range(self.n_windows):
            # Define center coords of left and right windows
            center_left = (self.left_x_curr, self.y_curr)
            center_right = (self.right_x_curr, self.y_curr)

            # Get x and y coords of pixels that fall within the window
            left_window_x, left_window_y = self.find_pixels(center_left)
            if draw_boxes == True:
                cv2.rectangle(self.out_img, self.top_left, self.bottom_right, color=(255, 0, 0), thickness=10)

            right_window_x, right_window_y = self.find_pixels(center_right)
            if draw_boxes == True:
                cv2.rectangle(self.out_img, self.top_left, self.bottom_right, color=(255, 255, 0), thickness=10)

            # Add pixels detected in window to master lists
            left_x.extend(left_window_x)
            left_y.extend(left_window_y)
            right_x.extend(right_window_x)
            right_y.extend(right_window_y)

            # Update pointers
            if len(left_window_x) >= self.min_pixels:
                self.left_x_curr = np.int32(np.mean(left_window_x))
            if len(right_window_x) >= self.min_pixels:
                self.right_x_curr = np.int32(np.mean(right_window_x))
            self.y_curr -= self.window_height

        return self.draw_lines(left_x, left_y, right_x, right_y)