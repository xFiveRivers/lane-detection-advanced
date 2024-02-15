import cv2
import numpy as np

class Lines():

    def __init__(self):
        self.img = None
        self.margin = 100
        self.min_pixels = 50
        self.n_windows = 8
        self.non_zero = None
        self.non_zero_x = None
        self.non_zero_y = None
        self.window_height = None
        self.left_x_curr = None
        self.right_x_curr = None
        self.y_curr = None

        self.left_fit = None
        self.right_fit = None

        self.top_left = None
        self.bottom_right = None
        self.out_img = None



    def draw_lines(self, left_x, left_y, right_x, right_y):
        
        # Introduce short-term memory
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

        # Get height of windows
        self.window_height = int(self.img.shape[0] // self.n_windows)

        # Create historgram
        y_cutoff = int(self.img.shape[0] * (1 - (1 / 3)))
        hist = np.sum(self.img[y_cutoff:, :], axis=0)

        # Define midpoint of historgram
        midpoint = int(hist.shape[0] / 2)

        # Define base location of left and right lines based on the peak
        # of the histograms to the left and right of the midpoint
        # left_x_base = np.argmax(hist[:midpoint])
        # right_x_base = np.argmax(hist[midpoint:]) + midpoint
        left_x_base = np.argmax(hist[:500])
        right_x_base = np.argmax(hist[800:]) + 800

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