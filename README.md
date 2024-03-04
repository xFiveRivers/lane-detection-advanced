# **Lane Detection**

## **Motivation**

With the modern reliance of cars, driving has become something akin to second nature for many people. Yet the sheer complexity of driving is outstanding. Think about how many calculations you are performing when you drive. If there is a hill coming up, you know to shift into a lower gear for extra torque. If you are approaching a school zone, you know to drive slower and be more cautious. When a sharp corner is approaching, you know what a good braking point is. The point here is that **you** are in control of the vehicle and it is **you** who is making the decisions of where and how to move the vehicle.

Now have you ever driven a vehicle that has some degree of automation? This could as simple as lane-keep assist or adaptive cruise control to as complex as semi or fully autonomous driving. It's a difficult and nerve-racking experience when you first try it. You know the dangers and the complexity of driving, it's not easy to trust a computer with someone's life.

Having tried some of these technologies, I have developed a keen interest in their development. How complex are they? What methods are implemented to make them robust to varying road conditions? What scientific or mathematical principles are applied to develop creative solutions Having an undergraduate degree in automotive engineering and a Masters in data science, it feels as though I am naturally gravitated to this field.

So why lane detection? Of all the angles to approach autonomous vehicles, lane detection seemed to be the best entry. Many ADAS or fully-autonomous vehicles fundamentally rely being able to detect the lane that a vehicle is driving on. Lane-keep assist or self-steering vehicles needs to know where the it is in relation to the lane boundry to calculate the appropriate steering input. Lane departure warning systems need to know the bounds of the lane to warn the driver if they are leaving them. An autonomous vehicle will need the lane boundries of the lanes next to it as well when making lane changes. There are many options to move forward with after understanding the basics of lane detection.

## **The Project**

The project focuses solely on developing a **computer vision algorithm** to detect the lane lines in a given test run. As compared to a simple method such as `Canny Edge Detection`, I wanted to try using a more complex solution that includes `perspective transformation` to bird's-eye view, colour spaces for `image thresholding`, and a `sliding window` algorithm to fit a second-degree polynomial for each lane line. The idea is to develop a solution that is more robust to varying road conditions as compared to more naive solutions.

Though, increasing robustness and generalization is no easy task with computer vision alone. Even sophisticated algorithms will need heavy calibrations to account for varying road conditions. The best algorithms right now, however, use deep learning algorithms and data that include high degrees of road condition variability.

