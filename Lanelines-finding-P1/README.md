# **Finding Lane Lines on the Road**

### The goals or steps of this project are the following:
* Make a pipeline that finds lane lines on the road.
* Reflect on my work in a written report.

### Reflection

### 1. Describe my pipeline

* My pipeline function named **find_line()** consisted 6 steps.

  1. Convert the images to grayscale;

  2. Use gaussian blur to reduce the noices;

  3. Use canny algorithm to get line edges;

  4. Select the interest region using cv2.fillPoly();

  5. Use HoughLineP to get the left and right lines;
  
  6. Draw the final detected left and right lane line on the original image.

* In order to draw the a single line on the left and right lines, I modified the **draw_lines()** function by the following steps:

 1. Calculate the left and right lines slope;

 2. Delete the maximum and minimum slopes and responding lines;

 3. Calculate average slope of left and right line; 

 4. Select two lines that whose slope are nearest to the average slope using a while loop, which delete line with maximum bias and calculate the average slope of the rest lines in every loop. 

 5. Caculate the median point of left and right line, and use the average slope to predict the left and right bottom point and the common top point.

 6. If the line slope of current frame, has  bias relativity bigger than the threshold compared to the pre-frame, then use pre-frame lines point instead to draw lines. 

* The following image is the result of my pipeline.


[//]: # (Image References)

[image1]: ./test_images_out/solidYellowCurve_out.jpg "Lane line"

![The detected lane line][image1]


### 2. The potential shortcomings of my current pipeline

* One potential shortcoming would be what would happen when the detected hough lines of left or right is empty, then code error maybe happen.

* Another potential shortcoming is that the threshold which decide whether use the pre-frame line points to replace the current line points to draw lines or not, maynot applicable to all video image frame. 



### 3. Possible improvement to my pipeline.

* A possible improvement would be to tune the canny and HoughLineP parameters and the threshold to get better result.


* Another potential improvement could be to save more then one pre-frame such as 5, and then use the average slope of these frame lines to draw lines. 