# **Lane Detection with Computer Vision**

## **Summary**

This repo is a personal project to explore autonomous vehicle concepts by creating a **computer vision** pipeline to detect the lane lines in front of a vehicle. Below is an example of the output from the pipeline.

> **Note:** The windows from the line-fitting algorithm are only shown here for conceptual purposes.

![Output Gif](https://github.com/xFiveRivers/lane-detection-advanced/blob/main/book_images/project_video.gif)

First we apply `camera distortion correction` to correct the image proportions. Then a `perspective transformation` is applied to the front-view image to create a top-down view of the lane ahead of the vehicle. `Thresholding` is then applied to separate the hot-pixels of the lane lines from the rest of the image. Finally, a `sliding-window` algorithm is applied to calculate a second-degree polynomial to fit the left and right lane lines.

I've also developed `jupyter` notebooks in which I describe the background and motivation for each step in the pipeline. My intention was to create a tutorial of sorts for myself, mostly to easily review the reasoning and code behind each method. It's also a good way of visually showing each step in the pipeline without having to decipher scripts.

## **Data**

All test and calibration data has been sourced from `Udacity's` Advanced Lane Lines [repository](https://github.com/udacity/CarND-Advanced-Lane-Lines).

## **Installation**

All of the packages required to run the pipeline are included in the `environment.yaml` file. To install the `conda` environment navigate to the root folder and run the following command:

```conda env create -f environment.yaml```

## **Usage**

The pipeline uses `docopt` to run the script through CLI and can be run using the following command in the root folder:

```python main.py <media> <input_path> <output_path>```

Where `<media>` specifies the type of media to process as a **string** (`'image'` or `'video'`), `<input_path>` specifies the path to the source file, and `<out_path>` specifies the path to write the processed file to. As an example:

```python main.py 'image' 'test_images/test1.jpg' 'output_media/test1.jpg'```

> **Note:** Video files must be saved as `.avi` files.

## **Motivation**

With the modern reliance of cars, driving has become something akin to second nature for many people. Yet the sheer complexity of driving is outstanding. Think about how many calculations you are performing when you drive. If there is a hill coming up, you know to shift into a lower gear for extra torque. If you are approaching a school zone, you know to drive slower and be more cautious. When a sharp corner is approaching, you know what a good braking point is. The point here is that **you** are in control of the vehicle and it is **you** who is making the decisions of where and how to move the vehicle.

Now have you ever driven a vehicle that has some degree of automation? This could as simple as lane-keep assist or adaptive cruise control to as complex as semi or fully autonomous driving. It's a difficult and nerve-racking experience when you first try it. You know the dangers and the complexity of driving, it's not easy to trust a computer with someone's life.

Having tried some of these technologies, I have developed a keen interest in their development. How complex are they? What methods are implemented to make them robust to varying road conditions? What scientific or mathematical principles are applied to develop creative solutions? Having an undergraduate degree in automotive engineering and a Masters in data science, it feels as though I am naturally gravitated to this field.

So why lane detection? Of all the angles to approach autonomous vehicles, lane detection seemed to be the best entry. Many ADAS or fully-autonomous vehicles fundamentally rely being able to detect the lane that a vehicle is driving in. Lane-keep assist or self-steering vehicles needs to know where the vehicle is in relation to the lane boundary to calculate the appropriate steering input. Lane departure warning systems need to know the bounds of the lane to warn the driver if they are leaving them. An autonomous vehicle will need the lane boundaries of the lanes next to it as well when making lane changes. There are many options to move forward with after understanding the basics of lane detection.

## **The Project**

The project focuses solely on developing a **computer vision algorithm** to detect the lane lines in a given test run. As compared to a simple method such as `Canny Edge Detection`, I wanted to try using a more complex solution that includes `perspective transformation` to bird's-eye view, colour spaces for `image thresholding`, and a `sliding window` algorithm to fit a second-degree polynomial for each lane line. The idea is to develop a solution that is more robust to varying road conditions as compared to more naive solutions.

Though, increasing robustness and generalization is no easy task with computer vision alone. Even sophisticated algorithms will need heavy calibrations to account for varying road conditions. The method I used is **not** a sophisticated algorithm. It has difficulty with dark or heavily shaded images and is susceptible to noise from debris or other markings on the road. The best algorithms right now, however, use deep learning algorithms and data that include high degrees of road condition variability.

**Author:** [Vikram Grewal](https://github.com/xFiveRivers)