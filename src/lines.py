import cv2
import numpy as np

class lines():

    def __init__(self):
        self.img = None
        self.margin = 100
        self.min_pixels = 50
        self.n_windows = 9
        self.non_zero = None
        self.non_zero_x = None
        self.non_zero_y = None
        self.window_height = None
        self.left_x_curr = None
        self.right_x_curr = None
        self.y_curr = None


    def draw_lines(self, left_x, left_y, right_x, right_y):
        
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        draw_x = np.linspace(0, self.img.shape[1], self.img.shape[1])

        draw_y_left = np.polyval(left_fit, draw_x)
        draw_y_right = np.polyval(right_fit, draw_x)

        draw_points_left = (np.asarray([draw_y_left, draw_x]).T).astype(np.int32)
        draw_points_right = (np.asarray([draw_y_right, draw_x]).T).astype(np.int32)

        out_img = np.zeros((self.img.shape[0], self.img.shape[1], 3))

        cv2.polylines(out_img, [draw_points_left], False, (0, 255, 0), thickness=8)
        cv2.polylines(out_img, [draw_points_right], False, (0, 255, 0), thickness=8)
        
        return out_img


    def get_features(self):

        # Create historgram
        hist = np.sum(self.img[self.img.shape[0] // 2:, :], axis=0)

        # Define midpoint of historgram
        midpoint = int(hist.shape[0] / 2)

        # Define base location of left and right lines based on the peak
        # of the histograms to the left and right of the midpoint
        left_x_base = np.argmax(hist[:midpoint])
        right_x_base = np.argmax(hist[midpoint:]) + midpoint

        # Create pointers for window center coordinates
        self.left_x_curr = left_x_base
        self.right_x_curr = right_x_base
        self.y_curr = self.img.shape[0] - self.window_height // 2

        # Get x and y poisitions of all non-zero pixels in image
        self.non_zero = self.img.nonzero()
        self.non_zero_x = np.array(self.non_zero[1])
        self.non_zero_y = np.array(self.non_zero[0])


    def find_pixels(self, center):
    
        # Define top-left and bottom-right x and y coords for window boundry
        top_left = (center[0] - self.margin, center[1] - (self.window_height // 2))
        bottom_right = (center[0] + self.margin, center[1] + self.window_height // 2)
        
        # Create conditions for array indexing
        cond_x = (self.non_zero_x >= top_left[0]) & (self.non_zero_x <= bottom_right[0])
        cond_y = (self.non_zero_y >= top_left[1]) & (self.non_zero_y <= bottom_right[1])

        # Index non-zero array to only take pixels within the window boundries
        targets_x = self.non_zero_x[cond_x & cond_y]
        targets_y = self.non_zero_y[cond_x & cond_y]

        return targets_x, targets_y
    

    def sliding_window(self, img):

        self.img = img

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
            right_window_x, right_window_y = self.find_pixels(center_right)

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